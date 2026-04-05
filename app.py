from flask import Flask,request
from generateRythm import generate_leaf_midi
from extractLeafVein import find_main_vein_endpoints, build_graph,extract_veins
from music import export_midi_to_wav

import cv2
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "music")

SOUNDFONT_PATH = os.path.abspath("FluidR3_GM.sf2")
SAMPLE_RATE    = 44100

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/img", methods=["PUT"])
def create_music():
    if "image" not in request.files:
        return "Error: no image sent.", 400

    file = request.files["image"]
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)
    leafImage = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(leafImage)

    vein = extract_veins(leafImage, enhanced)
    G = build_graph(leafImage, vein)
    # --- Replace with your actual getTopBottom call ---
    top, bottom = find_main_vein_endpoints(G,leafImage)

    generate_leaf_midi(G, top, bottom, output_path=f"{app.config['UPLOAD_FOLDER']}/leaf_drums.mid")
    export_midi_to_wav(
        midi_path    = f"{app.config['UPLOAD_FOLDER']}/leaf_drums.mid",
        output_path  = f"{app.config['UPLOAD_FOLDER']}/leaf_drums.wav",
        soundfont_path=SOUNDFONT_PATH,
        sample_rate=SAMPLE_RATE
    )
    return "completed"

@app.route("/")
def hello():
    print("h")
    return "hey"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)