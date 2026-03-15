# 🌿 Leaf Vein Music

Turn a photo of a leaf into music — by extracting its vein structure as a graph and sonifying it.

Inspired by [this YouTube video on leaf music theory](https://youtu.be/QRxDazqVquM?si=PuqlK8OXRgTI-avm) and the vein extraction approach from [LeafVeinExtraction](https://github.com/ZhengPeng7/LeafVeinExtraction).

---

## How it works

1. **Cut out the leaf** — the background is removed so only the leaf remains.
2. **Find the veins** — edge detection highlights the vein lines inside the leaf, then the leaf outline is subtracted so only the internal veins remain.
3. **Thin the veins** — the vein lines are reduced to 1 pixel wide so they can be traced cleanly.
4. **Build a map** — the vein skeleton is converted into a graph: branching points become nodes, and the segments connecting them become edges. Tiny noise fragments are removed, and any disconnected pieces are joined back together.
5. **Find the spine** — the two furthest-apart endpoints in the graph are identified as the tip and base of the main mid-rib. This spine is used to orient the leaf for music generation.
6. **Make music** — the graph is split into four quadrants around the spine. Each quadrant drives a different instrument (piano, bass, pad), the spine sets the drum rhythm, and the whole thing is rendered to MP3.

---

## Project structure

```
.
├── extractLeafVein.py   # Vein extraction + NetworkX graph construction
├── generate_rythm.py    # Graph → MIDI composition
├── music.py             # MIDI → WAV (fluidsynth) → EQ (sox)
├── app.py               # Flask API server
├── FluidR3_GM.sf2       # Soundfont (download separately, see below)
└── requirements.txt
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install system dependencies

**macOS**
```bash
brew install fluid-synth sox ffmpeg
```

**Ubuntu / Debian**
```bash
sudo apt install fluidsynth sox ffmpeg
```

### 3. Download the soundfont

The audio renderer needs a General MIDI soundfont. Download **FluidR3_GM.sf2** and place it in the project root:

```bash
# Direct download (Ubuntu/macOS)
wget https://keymusician01.s3.amazonaws.com/FluidR3_GM.zip
unzip FluidR3_GM.zip
```

Or find it via `apt`:
```bash
sudo apt install fluid-soundfont-gm
cp /usr/share/sounds/sf2/FluidR3_GM.sf2 .
```

---

## Running the API

```bash
python app.py
```

The server starts at `http://0.0.0.0:5000`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process` | Upload a leaf image → returns graph + preview + MP3 |
| `GET`  | `/health` | Health check |
| `GET`  | `/graph/<id>` | Retrieve cached graph JSON |
| `GET`  | `/preview/<id>` | Retrieve vein overlay image (base64 PNG) |
| `GET`  | `/audio/<id>` | Stream the generated MP3 |

### Example request

```bash
curl -X POST http://localhost:5000/process \
  -F "image=@leaf.jpg" | python -m json.tool
```

### Response shape

```json
{
  "session_id":     "abc-123",
  "graph":          { "nodes": [...], "edges": [...] },
  "preview_base64": "<base64 PNG>",
  "audio_base64":   "<base64 MP3>",
  "audio_mime":     "audio/mpeg",
  "stats": {
    "nodes": 142,
    "edges": 189,
    "processing_time_s": 8.4
  }
}
```

---

## Running standalone scripts

Generate MIDI directly from an image:

```bash
python generate_rythm.py
# Outputs: leaf_drums.mid
```

Render a MIDI file to MP3:

```bash
python music.py
# Outputs: leaf_music.wav
```

---

## Tips for best results

- Use a **flat, well-lit leaf** photographed against a plain background
- Higher contrast between leaf and background improves vein detection
- Larger leaves with prominent veins (e.g. monstera, maple, oak) produce richer graphs and more interesting music
- If the server returns `"Could not extract vein graph"`, try adjusting lighting or use a clearer photo

---

## Requirements

See `requirements.txt`. Key dependencies:

```
flask
flask-cors
opencv-python
scikit-image
networkx
scipy
mido
midiutil
sknw
```

System tools: `fluidsynth`, `sox`, `ffmpeg`
