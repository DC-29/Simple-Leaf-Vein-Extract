import modal
import io
import os
import base64
import sys

sys.path.insert(0, "/app")

# ─── Modal app + image ────────────────────────────────────────────────────────

app = modal.App("green-music")

# Build a container image with all dependencies pre-installed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "fluidsynth",
        "libfluidsynth-dev",
        "fluid-soundfont-gm",
        "wget",
    )
    .pip_install(
        "fastapi[standard]",
        "midiutil",
        "networkx",
        "numpy",
        "scikit-image",
        "sknw",
        "opencv-python-headless",
        "matplotlib",
        "scipy",
        "Pillow",
    )
    .add_local_file("generateRythm.py",   remote_path="/app/generateRythm.py")
    .add_local_file("extractLeafVein.py", remote_path="/app/extractLeafVein.py")
    .add_local_file("isolate_leaf.py",    remote_path="/app/isolate_leaf.py")
    .add_local_file("music.py",           remote_path="/app/music.py")
)

# Soundfont path inside the container (installed by fluid-soundfont-gm)
SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"


# ─── Endpoint ─────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    timeout=120,
    memory=1024,
)
@modal.web_endpoint(method="POST")
def generate_from_image(item: dict) -> dict:
    """
    Receive a base64-encoded leaf image, run the full pipeline,
    and return a base64-encoded WAV file.

    Request body:
    {
        "image_b64": "<base64 encoded PNG/JPG>",
        "scale_name": "major" | "minor" | null,   // optional
        "root_note": 60                            // optional MIDI note
    }

    Response:
    {
        "wav_b64": "<base64 encoded WAV>",
        "bpm": 45,
        "scale": "minor",
        "root_note": 60,
        "duration_seconds": 12.4
    }
    """
    import tempfile
    import subprocess
    import numpy as np
    import cv2

    from generateRythm import generate_leaf_midi
    from extractLeafVein import find_main_vein_endpoints, build_graph, extract_veins

    # ── 1. Decode image ──────────────────────────────────────────────────────
    image_bytes = base64.b64decode(item["image_b64"])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    leaf_image  = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    if leaf_image is None:
        return {"error": "Could not decode image"}

    # ── 2. Extract vein graph ────────────────────────────────────────────────
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(leaf_image)
    vein     = extract_veins(leaf_image, enhanced)
    G        = build_graph(leaf_image, vein)
    top, bottom = find_main_vein_endpoints(G, leaf_image)

    # ── 3. Generate MIDI ─────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        midi_path = os.path.join(tmp, "leaf.mid")
        wav_path  = os.path.join(tmp, "leaf.wav")

        result = generate_leaf_midi(
            G, top, bottom,
            output_path = midi_path,
            scale_name  = item.get("scale_name"),
            root_note   = item.get("root_note"),
        )

        # ── 4. Render MIDI → WAV with fluidsynth ─────────────────────────────
        subprocess.run([
            "fluidsynth",
            "-ni",
            "-g", "0.8",
            "-F", wav_path,
            "-r", "44100",
            SOUNDFONT,
            midi_path,
        ], check=True, capture_output=True)

        # ── 5. Encode WAV as base64 and return ───────────────────────────────
        with open(wav_path, "rb") as f:
            wav_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "wav_b64":    wav_b64,
        "bpm":        result["bpm"],
        "scale":      result["scale"],
        "root_note":  result["root_note"],
        "total_bars": result["total_bars"],
    }


# ─── Local test ───────────────────────────────────────────────────────────────

@app.local_entrypoint()
def test():
    """Quick local test — encode a leaf image and call the endpoint."""
    import base64

    image_path = "leavesImages/leaf2.png"
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("[test] Calling Modal function...")
    response = generate_from_image.remote({"image_b64": image_b64})

    if "error" in response:
        print(f"[test] Error: {response['error']}")
        return

    # Save returned WAV locally
    wav_bytes = base64.b64decode(response["wav_b64"])
    with open("test_output.wav", "wb") as f:
        f.write(wav_bytes)

    print(f"[test] BPM: {response['bpm']}, Scale: {response['scale']}")
    print(f"[test] WAV saved to test_output.wav")