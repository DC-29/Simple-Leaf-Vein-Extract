"""
Leaf Isolation
--------------
Isolates a leaf from any background, outputting a transparent PNG.

Usage:
    python isolate_leaf.py <input_image> [output_image]

Requirements:
    pip install opencv-python scikit-image numpy
"""

import sys
import numpy as np
import cv2
from pathlib import Path


def isolate_leaf(input_path: str, output_path: str | None = None) -> str:
    """
    Isolate a leaf from its background and save as a transparent PNG.

    Strategy:
      1. GrabCut for rough foreground/background separation
      2. HSV green-channel mask to catch leaf pixels GrabCut may miss
      3. Bilateral filter + edge-aware refinement for a clean boundary
      4. Keep only the largest connected component (the leaf)
      5. Feather the mask edges so the cutout looks natural
    """
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.with_stem(p.stem + "_isolated").with_suffix(".png"))

    # ── Load ─────────────────────────────────────────────────────────────────
    bgr = cv2.imread(input_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot load image: {input_path}")
    h, w = bgr.shape[:2]
    print(f"Loaded {input_path}  ({w}×{h})")

    # ── Step 1: GrabCut ───────────────────────────────────────────────────────
    print("Step 1/5 : GrabCut segmentation …")
    gc_mask = np.zeros((h, w), np.uint8)
    margin_x, margin_y = max(w // 12, 10), max(h // 12, 10)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc_mask, rect, bgd_model, fgd_model,
                iterCount=8, mode=cv2.GC_INIT_WITH_RECT)
    gc_fg = np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])

    # ── Step 2: HSV green mask ────────────────────────────────────────────────
    print("Step 2/5 : HSV colour refinement …")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Broad green/yellow-green range — covers most leaf colours
    green_mask = cv2.inRange(hsv, (20, 15, 15), (100, 255, 255)).astype(bool)

    # Combine: a pixel is leaf if GrabCut OR colour mask says so
    combined = (gc_fg | green_mask).astype(np.uint8)

    # ── Step 3: Morphological clean-up ───────────────────────────────────────
    print("Step 3/5 : Cleaning mask …")
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel_open)

    # ── Step 4: Largest connected component ──────────────────────────────────
    print("Step 4/5 : Keeping largest component …")
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    if n_labels > 1:
        # stats[:,4] = area; skip label 0 (background)
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        combined = (labels == biggest).astype(np.uint8)

    # Fill holes inside the leaf outline
    flood = combined.copy()
    cv2.floodFill(flood, None, (0, 0), 1)          # flood background → 1
    holes = (flood == 0).astype(np.uint8)           # unflooded = holes
    combined = np.clip(combined + holes, 0, 1)

    # ── Step 5: Feather edges for a natural look ─────────────────────────────
    print("Step 5/5 : Feathering edges …")
    alpha_f32 = combined.astype(np.float32)

    # Distance transform gives soft falloff near edges
    dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    feather_px = max(min(w, h) // 80, 3)           # scale feather to image size
    dist_norm = np.clip(dist / feather_px, 0.0, 1.0)

    # Smooth the raw mask boundary
    alpha_blur = cv2.GaussianBlur(alpha_f32, (7, 7), 0)
    alpha_final = np.where(combined == 1,
                           np.maximum(alpha_blur, dist_norm),
                           alpha_blur)
    alpha_u8 = (np.clip(alpha_final, 0, 1) * 255).astype(np.uint8)

    # ── Compose RGBA output ───────────────────────────────────────────────────
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_u8
    cv2.imwrite(output_path, rgba)

    leaf_px = int((alpha_u8 > 127).sum())
    total_px = h * w
    print(f"\nDone!  Leaf covers {leaf_px:,} / {total_px:,} px "
          f"({100 * leaf_px / total_px:.1f}% of frame)")
    print(f"Saved → {output_path}")
    return output_path


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python isolate_leaf.py <input_image> [output_image]")
        print("Example: python isolate_leaf.py leaf.jpg")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    isolate_leaf(input_file, output_file)