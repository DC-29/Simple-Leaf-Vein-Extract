"""
Leaf Vein Music API
-------------------
Flask backend that wraps the leaf vein extraction + music generation pipeline.

Endpoints:
  POST /process     - Upload leaf image ->   returns graph + audio as JSON
  GET  /health      - Health check
  GET  /graph/<id>  - Get cached graph by session ID
  GET  /preview/<id>- Get vein preview image by session ID

Install:
  pip install flask flask-cors opencv-python scikit-image networkx scipy mido sknw

Run:
  python app.py
"""

import os
import uuid
import base64
import tempfile
import traceback

import cv2
import mido
import numpy as np
import networkx as nx
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from generateGraph import extract_veins, build_graph

# ----------------------------------------------
# App setup
# ----------------------------------------------

app = Flask(__name__)
CORS(app)  # allow React Native / browser to call this API

UPLOAD_FOLDER = tempfile.mkdtemp()   # temp dir for uploads + outputs
sessions = {}                         # in-memory session cache: id -> data

# ----------------------------------------------
# Helpers
# ----------------------------------------------
def graph_to_json(G, img_h, img_w):
    """Serialize networkx graph to JSON-safe dict for the mobile app."""
    nodes = []
    for n in G.nodes():
        pos = G.nodes[n]['o']
        nodes.append({
            'id':     int(n),
            'x':      float(pos[1]) / img_w,   # normalize 0-1
            'y':      float(pos[0]) / img_h,
            'degree': int(G.degree(n)),
            'is_junction': G.degree(n) >= 3,
        })

    edges = []
    for s, e in G.edges():
        pts = G[s][e]['pts']
        # Downsample points for smaller payload (every 3rd point)
        sampled = pts[::3].tolist()
        edges.append({
            'source': int(s),
            'target': int(e),
            'length': len(pts),
            'points': [[float(r) / img_h, float(c) / img_w]
                       for r, c in sampled],
        })

    return {
        'nodes':     nodes,
        'edges':     edges,
        'img_width':  img_w,
        'img_height': img_h,
    }



def generate_preview(leafImage, G):
    """Render vein graph on leaf image, return as base64 PNG."""
    preview = cv2.cvtColor(leafImage, cv2.COLOR_GRAY2BGR)
    degrees = dict(G.degree())

    for s, e in G.edges():
        pts = G[s][e]['pts']
        for i in range(len(pts) - 1):
            p1 = (int(pts[i][1]),   int(pts[i][0]))
            p2 = (int(pts[i+1][1]), int(pts[i+1][0]))
            cv2.line(preview, p1, p2, (0, 200, 80), 1)

    for n in G.nodes():
        pos = G.nodes[n]['o']
        cx, cy = int(pos[1]), int(pos[0])
        color = (0, 0, 255) if degrees[n] >= 3 else (255, 200, 0)
        cv2.circle(preview, (cx, cy), 3, color, -1)

    _, buf = cv2.imencode('.png', preview)
    return base64.b64encode(buf).decode('utf-8')


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Leaf Vein API is running'})


@app.route('/process', methods=['POST'])
def process():
    """
    POST /process
    Body: multipart/form-data with field 'image' (jpg/png file)
    Returns JSON:
    {
      session_id: str,
      graph: { nodes, edges, img_width, img_height },
      preview_base64: str,       # base64 PNG of vein overlay
      audio_url: str,            # GET /audio/<session_id>
      stats: { nodes, edges, processing_time_s }
    }
    """
    import time
    t0 = time.time()

    # ── Validate input ──
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # ── Save uploaded image ──
        session_id  = str(uuid.uuid4())
        img_path    = os.path.join(UPLOAD_FOLDER, f'{session_id}_input.png')


        file.save(img_path)

        # ── Load + enhance ──
        leafImage = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if leafImage is None:
            return jsonify({'error': 'Could not read image'}), 400

        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(leafImage)

        # ── Extract veins + build graph ──
        vein = extract_veins(leafImage, enhanced)
        G    = build_graph(leafImage, vein, visualize=False)

        if G.number_of_nodes() == 0:
            return jsonify({'error': 'Could not extract vein graph from image'}), 422

        img_h, img_w = leafImage.shape


        # ── Serialize graph for mobile ──
        graph_data = graph_to_json(G, img_h, img_w)

        # ── Vein preview image ──
        preview_b64 = generate_preview(leafImage, G)

        # ── Cache session ──
        sessions[session_id] = {
            'graph':    graph_data,
            'preview':  preview_b64,
        }

        elapsed = round(time.time() - t0, 2)

        return jsonify({
            'session_id':     session_id,
            'graph':          graph_data,
            'preview_base64': preview_b64,
            'stats': {
                'nodes':              G.number_of_nodes(),
                'edges':              G.number_of_edges(),
                'processing_time_s':  elapsed,
            }
        })

    except Exception as ex:
        traceback.print_exc()
        return jsonify({'error': str(ex)}), 500


@app.route('/graph/<session_id>', methods=['GET'])
def get_graph(session_id):
    """GET /graph/<session_id> — return cached graph JSON."""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(sessions[session_id]['graph'])



@app.route('/preview/<session_id>', methods=['GET'])
def get_preview(session_id):
    """GET /preview/<session_id> — return base64 preview image."""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'preview_base64': sessions[session_id]['preview']})


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  Leaf Vein Music API")
    print("=" * 50)
    print(f"  Upload folder: {UPLOAD_FOLDER}")
    print()
    print("  Endpoints:")
    print("  POST /process      - Upload leaf image")
    print("  GET  /health       - Health check")
    print("  GET  /graph/<id>   - Get graph data")
    print("  GET  /preview/<id> - Get vein preview")
    print()
    print("  Starting on http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)