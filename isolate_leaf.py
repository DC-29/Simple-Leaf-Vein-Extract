"""
extract_foreground.py
---------------------
Extracts a leaf from any background, outputting a transparent PNG.

Automatically selects the best strategy based on background type:
  - Bokeh / blurred background  → sharpness-guided GrabCut
  - Cluttered / sharp background → darkness + saturation GrabCut

Usage:
    python extract_foreground.py <input_image> [output_image]

Requirements:
    pip install opencv-python numpy
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# ── Background type detection ─────────────────────────────────────────────────

def is_bokeh_background(bgr: np.ndarray) -> bool:
    """
    Detect whether the background is bokeh (blurred).
    Compares sharpness (Laplacian variance) in the image border vs the centre.
    A bokeh image has a sharp centre and blurry border.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    bw = max(w // 8, 20)   # border width
    bh = max(h // 8, 20)

    border = np.concatenate([
        gray[:bh, :].ravel(),
        gray[-bh:, :].ravel(),
        gray[:, :bw].ravel(),
        gray[:, -bw:].ravel(),
    ])
    cx1, cy1 = w // 4, h // 4
    cx2, cy2 = 3 * w // 4, 3 * h // 4
    centre = gray[cy1:cy2, cx1:cx2].ravel()

    def laplacian_var(pixels):
        img = pixels.reshape(-1, 1).astype(np.float32)
        return img.var()

    border_sharp  = cv2.Laplacian(gray[:bh, :], cv2.CV_64F).var() + \
                    cv2.Laplacian(gray[-bh:, :], cv2.CV_64F).var()
    centre_sharp  = cv2.Laplacian(gray[cy1:cy2, cx1:cx2], cv2.CV_64F).var()

    ratio = centre_sharp / max(border_sharp, 1e-6)
    print(f"Sharpness ratio (centre/border): {ratio:.2f} "
          f"→ {'bokeh' if ratio > 2.5 else 'cluttered'} background detected")
    return ratio > 2.5


# ── Shared utilities ──────────────────────────────────────────────────────────

def run_grabcut(bgr: np.ndarray, margin_frac: float = 0.07,
                iters: int = 8) -> np.ndarray:
    """Run GrabCut and return a uint8 foreground mask (0 or 255)."""
    h, w = bgr.shape[:2]
    gc_mask = np.zeros((h, w), np.uint8)
    mx, my = int(w * margin_frac), int(h * margin_frac)
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc_mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255


def clean_and_largest(mask: np.ndarray,
                       close_px: int = 21, open_px: int = 5) -> np.ndarray:
    """Morphological close→open, keep largest component, fill holes."""
    k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    k_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_c)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_o)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if stats.shape[0] > 1:
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == biggest).astype(np.uint8) * 255

    # Fill enclosed holes
    flood = mask.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = (flood == 0).astype(np.uint8) * 255
    return cv2.bitwise_or(mask, holes)


def feather_and_save(bgr: np.ndarray, mask: np.ndarray,
                     output_path: str, preview_path: str):
    """Apply distance-transform feathering, save RGBA + white-bg preview."""
    h, w = bgr.shape[:2]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    feather_px = max(min(w, h) // 80, 3)
    alpha_f = np.clip(dist / feather_px, 0.0, 1.0).astype(np.float32)
    alpha_u8 = (cv2.GaussianBlur(alpha_f, (7, 7), 0) * 255).astype(np.uint8)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_u8
    cv2.imwrite(output_path, rgba)

    white = np.ones_like(bgr) * 255
    a = alpha_u8.astype(np.float32) / 255.0
    for c in range(3):
        white[:, :, c] = (bgr[:, :, c] * a + 255 * (1 - a)).astype(np.uint8)
    cv2.imwrite(preview_path, white)

    leaf_px = int((alpha_u8 > 127).sum())
    print(f"Leaf covers {leaf_px:,} / {h*w:,} px ({100*leaf_px/(h*w):.1f}%)")
    print(f"Transparent PNG  → {output_path}")
    print(f"White BG preview → {preview_path}")


# ── Strategy A: bokeh / blurred background ────────────────────────────────────

def extract_bokeh(bgr: np.ndarray) -> np.ndarray:
    """
    Exploit the sharpness difference between the in-focus leaf and
    the out-of-focus bokeh background.
    """
    print("Strategy : bokeh background")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Sharpness map via Laplacian — leaf is sharp, bg is blurry
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F)).astype(np.float32)
    sharp_map = cv2.GaussianBlur(lap, (51, 51), 0)
    sharp_norm = cv2.normalize(sharp_map, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    _, sharp_mask = cv2.threshold(sharp_norm, 25, 255, cv2.THRESH_BINARY)

    # GrabCut for rough fg/bg split
    gc_fg = run_grabcut(bgr, margin_frac=0.06, iters=8)

    # AND: must be sharp AND GrabCut-foreground
    combined = cv2.bitwise_and(gc_fg, sharp_mask)

    # Also keep saturated pixels the sharpness check might miss
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, sat_mask = cv2.threshold(hsv[:, :, 1], 40, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(gc_fg, sat_mask))

    return clean_and_largest(combined, close_px=21, open_px=5)


# ── Strategy B: cluttered / sharp background ──────────────────────────────────

def extract_cluttered(bgr: np.ndarray) -> np.ndarray:
    """
    Use darkness + colour saturation cues alongside GrabCut for
    leaves against busy, sharp backgrounds.
    """
    print("Strategy : cluttered background")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # GrabCut
    gc_fg = run_grabcut(bgr, margin_frac=0.08, iters=10)

    # Darkness cue — leaf is darker than light walls/ceiling
    _, dark_mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

    # Saturation cue — leaf has colour vs greyscale background
    _, sat_mask = cv2.threshold(hsv[:, :, 1], 20, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(gc_fg, cv2.bitwise_or(dark_mask, sat_mask))
    return clean_and_largest(combined, close_px=25, open_px=7)


# ── Main entry ────────────────────────────────────────────────────────────────

def isolate_leaf(input_path: str, output_path: str | None = None) -> str:
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + "_fg").with_suffix(".png"))

    preview_path = str(
        Path(output_path).with_stem(Path(output_path).stem + "_preview")
    )

    bgr = cv2.imread(input_path)

    if bgr is None:
        raise FileNotFoundError(f"Cannot load: {input_path}")
    h, w = bgr.shape[:2]
    print(f"Loaded {input_path}  ({w}×{h})")

    if is_bokeh_background(bgr):
        mask = extract_bokeh(bgr)
    else:
        mask = extract_cluttered(bgr)

    feather_and_save(bgr, mask, output_path, preview_path)

    preview = Image.open(preview_path)
    preview.convert("RGB")
    preview.thumbnail((600,600), Image.LANCZOS)

    preview.save(preview_path)
    print(preview.size)

    return preview_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_foreground.py <input_image> [output_image]")
        sys.exit(1)

    print(
        isolate_leaf(
            sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else None    
        )
    )