import os
import sys
import time
import math
import threading
import pygame
import numpy as np
import mido
import cv2
from generateGraph import extract_veins, build_graph

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
WINDOW_W      = 1000
WINDOW_H      = 750
BG_COLOR      = (5, 5, 15)        # near-black background
IDLE_COLOR    = (30, 60, 40)      # dim green for inactive edges
GLOW_LAYERS   = 6                 # number of blur layers for glow effect
FPS           = 60
RECORD        = True              # Set False to disable recording
OUTPUT_VIDEO  = 'leaf_viz.mp4'


# ─────────────────────────────────────────────
# COLOR HELPERS
# ─────────────────────────────────────────────
def pitch_to_color(pitch):
    """Map MIDI pitch to a vivid HSV-style color."""
    hue = (pitch % 12) / 12.0      # 0-1 based on note in octave
    octave_norm = (pitch - 36) / 60.0  # brightness by octave
    octave_norm = max(0, min(1, octave_norm))

    # Convert hue to RGB (simple HSV with S=1, V=1)
    h = hue * 6
    i = int(h)
    f = h - i
    q = 1 - f
    p = 0.0

    rgb_map = [
        (1, f, p), (q, 1, p), (p, 1, f),
        (p, q, 1), (f, p, 1), (1, p, q)
    ]
    r, g, b = rgb_map[i % 6]

    # Scale brightness by octave
    brightness = 0.5 + 0.5 * octave_norm
    r = int(r * 255 * brightness)
    g = int(g * 255 * brightness)
    b = int(b * 255 * brightness)
    return (r, g, b)


def glow_color(base_color, layer, total_layers):
    """Fade color for glow layers — outer layers dimmer and wider."""
    fade = 1 - (layer / total_layers) ** 0.5
    return tuple(int(c * fade) for c in base_color)


# ─────────────────────────────────────────────
# COORDINATE MAPPING
# ─────────────────────────────────────────────
def map_pts(pts, img_h, img_w, win_w, win_h, margin=40):
    """Map image pixel coordinates to window coordinates."""
    scale = min((win_w - 2*margin) / img_w, (win_h - 2*margin) / img_h)
    offset_x = (win_w - img_w * scale) / 2
    offset_y = (win_h - img_h * scale) / 2
    mapped = []
    for r, c in pts:
        x = int(c * scale + offset_x)
        y = int(r * scale + offset_y)
        mapped.append((x, y))
    return mapped


def map_node(pos, img_h, img_w, win_w, win_h, margin=40):
    scale = min((win_w - 2*margin) / img_w, (win_h - 2*margin) / img_h)
    offset_x = (win_w - img_w * scale) / 2
    offset_y = (win_h - img_h * scale) / 2
    x = int(pos[1] * scale + offset_x)
    y = int(pos[0] * scale + offset_y)
    return (x, y)


# ─────────────────────────────────────────────
# DRAW HELPERS
# ─────────────────────────────────────────────
def draw_glow_line(surface, color, points, width=2):
    """Draw a line with a multi-layer glow effect."""
    if len(points) < 2:
        return
    for layer in range(GLOW_LAYERS, 0, -1):
        gc = glow_color(color, layer, GLOW_LAYERS)
        lw = width + layer * 2
        for i in range(len(points) - 1):
            pygame.draw.line(surface, gc, points[i], points[i+1], lw)
    # Bright core
    for i in range(len(points) - 1):
        pygame.draw.line(surface, color, points[i], points[i+1], width)


def draw_glow_circle(surface, color, pos, radius=6):
    """Draw a circle with glow."""
    for layer in range(GLOW_LAYERS, 0, -1):
        gc = glow_color(color, layer, GLOW_LAYERS)
        r = radius + layer * 2
        pygame.draw.circle(surface, gc, pos, r)
    pygame.draw.circle(surface, color, pos, radius)


# ─────────────────────────────────────────────
# ACTIVE EDGE TRACKER
# ─────────────────────────────────────────────
class ActiveEdge:
    def __init__(self, pts, color, duration_ms):
        self.pts       = pts
        self.color     = color
        self.start_ms  = pygame.time.get_ticks()
        self.duration  = duration_ms
        self.progress  = 0.0   # 0 = just started, 1 = finished

    def update(self):
        elapsed = pygame.time.get_ticks() - self.start_ms
        self.progress = min(1.0, elapsed / self.duration)

    @property
    def alive(self):
        return self.progress < 1.0

    def draw(self, surface):
        if not self.pts:
            return
        # Draw only up to current progress point (traveling effect)
        n = max(2, int(len(self.pts) * self.progress))
        visible_pts = self.pts[:n]

        # Fade out near end
        fade = 1.0 - max(0, (self.progress - 0.7) / 0.3)
        color = tuple(int(c * fade) for c in self.color)

        draw_glow_line(surface, color, visible_pts, width=2)

        # Bright travelling dot at the front
        if visible_pts:
            draw_glow_circle(surface, self.color, visible_pts[-1], radius=4)


# ─────────────────────────────────────────────
# MIDI EVENT SCHEDULER
# ─────────────────────────────────────────────
def parse_midi_events(midi_file):
    """Parse MIDI into timed note_on events."""
    mid = mido.MidiFile(midi_file)
    tempo = 500000  # default 120 BPM
    events = []
    abs_time_s = 0.0

    for msg in mid.tracks[0]:
        delta_s = mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        abs_time_s += delta_s
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif msg.type == 'note_on' and msg.velocity > 0:
            events.append({
                'time_s': abs_time_s,
                'note':   msg.note,
                'vel':    msg.velocity,
            })

    print(f"Parsed {len(events)} note events, duration {abs_time_s:.1f}s")
    return events


# ─────────────────────────────────────────────
# MAIN VISUALIZER
# ─────────────────────────────────────────────
def run_visualizer(G, leafImage, midi_file='leaf_music.mid'):

    img_h, img_w = leafImage.shape
    degrees = dict(G.degree())

    # Pre-map all edge point lists to window coords
    edge_win_pts = {}
    for s, e in G.edges():
        raw_pts = G[s][e]['pts']
        edge_win_pts[(s, e)] = map_pts(raw_pts, img_h, img_w, WINDOW_W, WINDOW_H)
        edge_win_pts[(e, s)] = edge_win_pts[(s, e)]

    # Pre-map node positions
    node_win_pos = {}
    for n in G.nodes():
        node_win_pos[n] = map_node(G.nodes[n]['o'], img_h, img_w, WINDOW_W, WINDOW_H)

    # Build BFS order matching generateMusic.py traversal
    nodes_pos = {n: G.nodes[n]['o'] for n in G.nodes()}
    start_node = min(G.nodes(), key=lambda n: nodes_pos[n][0])

    bfs_edges = []   # [(node, neighbor), ...]
    visited_edges = set()
    visited_nodes = {start_node}
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        for neighbor in G.neighbors(node):
            ek = tuple(sorted([node, neighbor]))
            if ek not in visited_edges:
                visited_edges.add(ek)
                bfs_edges.append((node, neighbor))
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)

    # Parse MIDI events
    midi_events = parse_midi_events(midi_file)

    # Pair BFS edges with MIDI events (same order as generateMusic.py)
    paired = list(zip(bfs_edges, midi_events[:len(bfs_edges)]))

    # ── pygame setup ──
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Leaf Vein Music Visualizer")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('monospace', 14)

    # ── Video recorder setup ──
    video_writer = None
    if RECORD:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WINDOW_W, WINDOW_H))
        print(f"Recording to {OUTPUT_VIDEO}...")

    # Static background surface (idle graph)
    bg_surface = pygame.Surface((WINDOW_W, WINDOW_H))
    bg_surface.fill(BG_COLOR)
    for s, e in G.edges():
        pts = edge_win_pts[(s, e)]
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                pygame.draw.line(bg_surface, IDLE_COLOR, pts[i], pts[i+1], 1)

    # ── MIDI playback thread ──
    active_edges = []
    play_start   = None

    def midi_thread():
        nonlocal play_start
        try:
            pygame.mixer.init(44100, -16, 2, 512)
            pygame.mixer.music.load(midi_file)  # works for both .wav and .mid
            pygame.mixer.music.play()
            print("Audio playing...")
        except Exception as ex:
            print(f"Audio playback error: {ex}")
        play_start = pygame.time.get_ticks()

    t = threading.Thread(target=midi_thread, daemon=True)
    t.start()

    # Wait for play to start
    while play_start is None:
        time.sleep(0.01)

    event_idx  = 0
    running    = True
    note_count = 0

    while running:
        now_ms   = pygame.time.get_ticks()
        elapsed_s = (now_ms - play_start) / 1000.0

        # ── handle quit ──
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

        # ── trigger new edges ──
        while event_idx < len(paired):
            (node, neighbor), midi_ev = paired[event_idx]
            if elapsed_s >= midi_ev['time_s']:
                pitch    = midi_ev['note']
                color    = pitch_to_color(pitch)
                pts      = edge_win_pts.get((node, neighbor), [])
                # duration proportional to edge length
                duration = max(300, len(pts) * 15)
                active_edges.append(ActiveEdge(pts, color, duration))
                note_count += 1
                event_idx  += 1
            else:
                break

        # ── update active edges ──
        for ae in active_edges:
            ae.update()
        active_edges = [ae for ae in active_edges if ae.alive]

        # ── draw ──
        screen.blit(bg_surface, (0, 0))

        # Draw active edges
        for ae in active_edges:
            ae.draw(screen)

        # Draw all nodes (dim)
        for n, pos in node_win_pos.items():
            r = 3 if degrees[n] >= 3 else 2
            pygame.draw.circle(screen, (40, 80, 50), pos, r)

        # HUD
        hud = font.render(
            f"Notes: {note_count}  Active: {len(active_edges)}  "
            f"Time: {elapsed_s:.1f}s   ESC to quit",
            True, (100, 180, 120)
        )
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

        # ── Capture frame for recording ──
        if RECORD and video_writer:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))       # (w,h,c) → (h,w,c)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        # Stop when all events done and no active edges
        if event_idx >= len(paired) and not active_edges:
            time.sleep(1)
            running = False

    if RECORD and video_writer:
        video_writer.release()
        print(f"Video saved: {OUTPUT_VIDEO}")

    pygame.quit()
    print("Visualizer closed.")


# ─────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────
def midi_to_wav_python(midi_file, output_wav='leaf_music.wav', sample_rate=44100):
    """Convert MIDI to WAV using pure Python - no soundfont needed."""
    import pretty_midi
    import scipy.io.wavfile as wav
    import numpy as np

    print("Synthesizing audio from MIDI (no soundfont needed)...")
    pm = pretty_midi.PrettyMIDI(midi_file)

    # pretty_midi can synthesize audio directly
    audio = pm.fluidsynth(fs=sample_rate)  # uses built-in samples

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    audio_int = (audio * 32767).astype(np.int16)

    wav.write(output_wav, sample_rate, audio_int)
    print(f"WAV saved: {output_wav}")
    return output_wav


def merge_audio_video(video_file, audio_file, output_file='leaf_final.mp4'):
    """Merge video + audio using ffmpeg."""
    import subprocess
    print(f"Merging video + audio → {output_file}")
    result = subprocess.run([
        'ffmpeg', '-y',
        '-i', video_file,
        '-i', audio_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_file
    ], capture_output=True, text=True)
    if os.path.exists(output_file):
        print(f"Final video with audio saved: {output_file}")
        return True
    else:
        print(f"ffmpeg merge failed: {result.stderr}")
        print("Install ffmpeg from https://ffmpeg.org/download.html")
        return False


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    image_fileName = 'leaf1.png'
    if len(sys.argv) > 1:
        image_fileName = sys.argv[1]

    image_path = f'leavesImages/{image_fileName}'
    midi_file  = 'leaf_music.mid'

    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        sys.exit(1)

    if not os.path.exists(midi_file):
        print(f"Error: '{midi_file}' not found. Run generateMusic.py first.")
        sys.exit(1)

    # Load and process leaf
    leafImage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(leafImage)

    vein = extract_veins(leafImage, enhanced)
    G    = build_graph(leafImage, vein, visualize=False)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Convert MIDI to WAV for audio playback + merging
    wav_file = midi_to_wav_python(midi_file)

    print("Starting visualizer...")
    run_visualizer(G, leafImage, wav_file if wav_file else midi_file)

    # After visualizer closes, merge video + audio
    if RECORD and wav_file and os.path.exists(OUTPUT_VIDEO):
        merge_audio_video(OUTPUT_VIDEO, wav_file, output_file='leaf_final.mp4')
        print("Done! Open leaf_final.mp4 to watch with sound.")