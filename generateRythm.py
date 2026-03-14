import math
import random
import networkx as nx
import numpy as np
from midiutil import MIDIFile
 
 
# ─── Scales ───────────────────────────────────────────────────────────────────
 
SCALES = {
    "major":       [0, 2, 4, 5, 7, 9, 11],
    "minor":       [0, 2, 3, 5, 7, 8, 10],
    "dorian":      [0, 2, 3, 5, 7, 9, 10],
    "phrygian":    [0, 1, 3, 5, 7, 8, 10],
    "lydian":      [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":  [0, 2, 4, 5, 7, 9, 10],
    "pentatonic":  [0, 2, 4, 7, 9],
}
 
# General MIDI program numbers (0-indexed)
GM_PIANO   = 0    # Acoustic Grand Piano
GM_BASS    = 32   # Acoustic Bass
GM_PAD     = 89   # Pad 2 (warm)
 
# MIDI channels (0-indexed, channel 9 reserved for drums)
CH_PIANO   = 0
CH_BASS    = 1
CH_PAD     = 2
CH_DRUMS   = 9
 
# Tracks
TR_DRUMS   = 0
TR_PIANO   = 1
TR_BASS    = 2
TR_PAD     = 3
 
KICK_NOTE  = 36

# ─── Chord helpers ────────────────────────────────────────────────────────────

TRIAD_INTERVALS = {
    "major": [0, 4, 7],    # major triad intervals above root
    "minor": [0, 3, 7],    # minor triad intervals above root
}

# Fixed progressions — (semitone offset from root_note, triad quality)
PROGRESSIONS = {
    "major": [
        (0,  "major"),   # I
        (5,  "major"),   # IV
        (7,  "major"),   # V
        (9,  "minor"),   # vi
    ],
    "minor": [
        (0,  "minor"),   # i
        (10, "major"),   # VII
        (8,  "major"),   # VI
        (7,  "minor"),   # v
    ],
}


def semitone_distance(a: int, b: int) -> int:
    """Minimum semitone distance between two notes, ignoring octave."""
    diff = abs((a % 12) - (b % 12))
    return min(diff, 12 - diff)


def choose_chord(melody_note: int, root_note: int, scale_name: str) -> tuple:
    """
    Pick the chord from the fixed progression whose root is closest
    in pitch to the melody note. Returns (chord_root, triad_type).
    Chord is voiced in the octave just below the melody note.
    """
    key = "major" if scale_name == "major" else "minor"
    best_offset, best_type = min(
        PROGRESSIONS[key],
        key=lambda ct: semitone_distance(melody_note, root_note + ct[0])
    )
    chord_root = root_note + best_offset
    # Place chord root just below melody note
    while chord_root >= melody_note:
        chord_root -= 12
    while chord_root < melody_note - 12:
        chord_root += 12
    return chord_root, best_type


def build_chord(melody_note: int, root_note: int, scale_name: str) -> list:
    """
    Return 3 MIDI notes forming the closest-match triad from the fixed
    progression, voiced below the melody note.
    """
    chord_root, triad_type = choose_chord(melody_note, root_note, scale_name)
    return [max(0, min(127, chord_root + i)) for i in TRIAD_INTERVALS[triad_type]]



def get_node_position(G: nx.Graph, node: int) -> np.ndarray:
    """Return [y, x] position of a node as float array."""
    return np.array(G.nodes[node]['o'], dtype=float)
 
 
def edge_euclidean_length(G: nx.Graph, u: int, v: int) -> float:
    return float(np.linalg.norm(get_node_position(G, u) - get_node_position(G, v)))
 
 
def round_to_step(value: float, step: float = 0.25) -> float:
    return max(step, round(value / step) * step)
 
 
def build_spine_axes(top: list, bottom: list):
    """
    Return (spine_unit, perp_unit, bottom_arr) where:
      spine_unit : unit vector from bottom → top
      perp_unit  : unit vector perpendicular to spine (pointing 'right')
    """
    bottom_arr = np.array(bottom, dtype=float)
    top_arr    = np.array(top,    dtype=float)
    spine_vec  = top_arr - bottom_arr
    spine_len  = np.linalg.norm(spine_vec)
    if spine_len == 0:
        spine_unit = np.array([1.0, 0.0])
    else:
        spine_unit = spine_vec / spine_len
    # Perpendicular: rotate spine_unit 90° clockwise → (dy, -dx) → (y→x swap)
    perp_unit = np.array([spine_unit[1], -spine_unit[0]])
    return spine_unit, perp_unit, bottom_arr
 
 
def project_node(pos: np.ndarray, bottom_arr: np.ndarray,
                 spine_unit: np.ndarray, perp_unit: np.ndarray):
    """
    Return (spine_proj, perp_proj) — position in the rotated leaf frame.
    spine_proj > 0 means closer to tip (top).
    perp_proj  > 0 means to the right of the spine.
    """
    delta = pos - bottom_arr
    return float(np.dot(delta, spine_unit)), float(np.dot(delta, perp_unit))
 
 
# ─── Duration normalisation (rank-based for variance) ────────────────────────
 
def normalize_durations(raw_lengths: list, beats_per_bar: float = 16.0,
                        min_step: float = 0.5) -> list:
    """
    Map raw edge lengths to durations with high variance using rank mapping.
    Shorter edges → shorter durations, longer → longer, with the full
    palette spread evenly across the rank range.
    """
    n = len(raw_lengths)
    if n == 0:
        return []
    if n == 1:
        return [beats_per_bar]
    

 
    palette = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4]
 
    sorted_idx = sorted(range(n), key=lambda i: raw_lengths[i])
    ranks = [0] * n
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
 
    def rank_to_dur(rank):
        t = rank / max(n - 1, 1)
        pidx = round(t * (len(palette) - 1))
        return palette[pidx]
 
    durations = [rank_to_dur(r) for r in ranks]
    acc = 0
    for i in range(len(durations)):
        acc += durations[i]
        if acc>beats_per_bar:
            durations[i-1] += durations[i] - (acc - beats_per_bar)
            durations = durations[:i]
            break
    return durations
 
 
def normalize_bpm(spine_length: float,
                  min_length: float = 450,  max_length: float = 500.0,
                  min_bpm:    int   = 30,    max_bpm:    int   = 60) -> int:
    clamped = max(min_length, min(max_length, spine_length))
    ratio   = (clamped - min_length) / (max_length - min_length)
    return int(round(max_bpm - ratio * (max_bpm - min_bpm)))
 
 
# ─── Spine detection ──────────────────────────────────────────────────────────
 
def get_spine_nodes(G: nx.Graph, top: list, bottom: list, padding: int = 10) -> list:
    x_min = int(min(top[1], bottom[1])) - padding
    x_max = int(max(top[1], bottom[1])) + padding
    y_min = int(min(top[0], bottom[0]))
    y_max = int(max(top[0], bottom[0]))
    
    degrees = G.degree
    result = []
    for n in G.nodes:
        pos = get_node_position(G, n)
        ny, nx_ = int(pos[0]), int(pos[1])
        if x_min <= nx_ <= x_max and y_min <= ny <= y_max and degrees[n]>3:
            result.append(n)
    return result
 
 
def order_spine_nodes(spine_nodes: list, G: nx.Graph,
                      spine_unit: np.ndarray, bottom_arr: np.ndarray) -> list:
    def proj(n):
        return float(np.dot(get_node_position(G, n) - bottom_arr, spine_unit))
    return sorted(spine_nodes, key=proj)
 
 
# ─── Quadrant classification ──────────────────────────────────────────────────
 
 
def get_quadrant_nodes(G: nx.Graph, spine_nodes_set: set,
                       bottom_arr: np.ndarray, spine_unit: np.ndarray,
                       perp_unit: np.ndarray, spine_length: float) -> dict:
    """Return dict mapping quadrant name → list of node IDs."""
 
    # Collect all non-spine nodes with degree > 1
    candidates = [
        n for n in G.nodes
        if n not in spine_nodes_set 
    ]
 
    if not candidates:
        print("[green_music] Warning: no candidate nodes found, using all non-spine nodes")
        candidates = [n for n in G.nodes if n not in spine_nodes_set]
 
    # Project all candidates onto both axes
    s_vals = []
    p_vals = []
    for n in candidates:
        pos = get_node_position(G, n)
        s, p = project_node(pos, bottom_arr, spine_unit, perp_unit)
        s_vals.append(s)
        p_vals.append(p)
 
    # Use median as split point — robust to skewed distributions
    s_mid = float(np.median(s_vals))
    p_mid = float(np.median(p_vals))
 
    print(f"[green_music] Spine split at s={s_mid:.1f}, perp split at p={p_mid:.1f}")
    print(f"[green_music] s range: [{min(s_vals):.1f}, {max(s_vals):.1f}], "
          f"p range: [{min(p_vals):.1f}, {max(p_vals):.1f}]")
 
    quadrants = {"top_left": [], "top_right": [], "bottom_left": [], "bottom_right": []}
    for n, s, p in zip(candidates, s_vals, p_vals):
        spine_side = "top"   if s > s_mid else "bottom"
        perp_side  = "right" if p > p_mid else "left"
        quadrants[f"{spine_side}_{perp_side}"].append(n)
 
    return quadrants
 
 
# ─── Scale / pitch helpers ────────────────────────────────────────────────────
 
def build_note_list(root: int, scale_intervals: list, octaves: int = 2) -> list:
    """
    Build a sorted list of MIDI note numbers spanning `octaves` octaves
    starting from `root`.
    e.g. root=60 (C4), major scale, 2 octaves → [60,62,64,65,67,69,71,72,74,76,77,79,81,83]
    """
    notes = []
    for oct_offset in range(octaves):
        for interval in scale_intervals:
            note = root + oct_offset * 12 + interval
            if note <= 127:
                notes.append(note)
    return sorted(set(notes))
 
 
def pos_to_pitch(perp_proj: float, perp_min: float, perp_max: float,
                 note_list: list) -> int:
    """Map a perpendicular projection value to a MIDI note from note_list."""
    if perp_max == perp_min:
        return note_list[len(note_list) // 2]
    t = (perp_proj - perp_min) / (perp_max - perp_min)
    t = max(0.0, min(1.0, t))
    idx = int(round(t * (len(note_list) - 1)))
    return note_list[idx]
 
 
# ─── Melody generation from a quadrant ───────────────────────────────────────
 
def generate_melody_for_quadrant(
    G: nx.Graph,
    nodes: list,
    note_list: list,
    bottom_arr: np.ndarray,
    spine_unit: np.ndarray,
    perp_unit: np.ndarray,
    min_step: float = 0.125,
) -> list:
    """
    BFS over the full graph starting from the quadrant's entry node.
    A note is emitted every time we visit a node that belongs to this quadrant.
    Duration is the Euclidean distance travelled since the last emitted note.
    Returns list of (pitch, duration) tuples.
    """
    if not nodes:
        return []
 
    node_set = set(nodes)
 
    # Pitch mapping range from quadrant nodes
    perp_projs = []
    for n in nodes:
        pos = get_node_position(G, n)
        _, p = project_node(pos, bottom_arr, spine_unit, perp_unit)
        perp_projs.append(p)
    perp_min = min(perp_projs)
    perp_max = max(perp_projs)
 
    # Start BFS from the quadrant node with the highest spine projection
    start = max(nodes, key=lambda n: float(
        np.dot(get_node_position(G, n) - bottom_arr, spine_unit)
    ))
 
    # Full-graph BFS — emit note only when landing on a quadrant node
    visited        = set()
    queue          = [start]
    events         = []
    last_pos       = get_node_position(G, start)
    dist_accum     = 0.0   # accumulate distance until next quadrant node
 
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
 
        curr_pos  = get_node_position(G, current)
        step_dist = float(np.linalg.norm(curr_pos - last_pos))
        dist_accum += step_dist
        last_pos   = curr_pos
 
        if current in node_set:
            # Emit a note for this quadrant node
            _, p  = project_node(curr_pos, bottom_arr, spine_unit, perp_unit)
            pitch = pos_to_pitch(p, perp_min, perp_max, note_list)
 
            # Convert accumulated pixel distance to beats
            duration = round_to_step(dist_accum / 200.0, min_step)
            duration = max(min_step, min(2.0, duration))
 
            events.append((pitch, duration))
            dist_accum = 0.0   # reset accumulator after each emitted note
 
        for nb in G.neighbors(current):
            if nb not in visited:
                queue.append(nb)
 
    return events
 
 
# ─── Master generation function ───────────────────────────────────────────────
 
def generate_leaf_midi(
    G: nx.Graph,
    top: list,
    bottom: list,
    output_path: str = "leaf_music.mid",
    beats_per_bar: int = 16,
    drum_bars: int = 4,
    spine_padding: int = 10,
    velocity: int = 90,
    scale_name: str = None,    # None = random
    root_note: int = None,     # None = random (C3–G4)
) -> dict:
    """
    Generate a full leaf MIDI composition:
      - Drums  : mid-ridge rhythm, looped for drum_bars bars
      - Piano  : top-left + top-right quadrant melody (2 octaves)
      - Bass   : bottom-left quadrant, lower octave
      - Pad    : bottom-right quadrant, pad/strings voicing
 
    Parameters
    ----------
    G             : NetworkX graph from sknw.build_sknw
    top           : [y, x] tip of the mid-ridge
    bottom        : [y, x] base of the mid-ridge
    output_path   : output .mid file path
    beats_per_bar : 4 for 4/4
    drum_bars     : how many bars the drum pattern loops for
    spine_padding : ±x pixel width for spine bounding box
    velocity      : base MIDI velocity
    scale_name    : key in SCALES dict, or None for random
    root_note     : MIDI root note, or None for random
    """
 
    # ── 0. Pick scale and root ──────────────────────────────────────────────
    if scale_name is None:
        scale_name = random.choice(list(SCALES.keys()))
    if root_note is None:
        root_note = random.randint(48, 67)   # C3–G4
 
    scale_intervals = SCALES[scale_name]
    print(f"[green_music] Scale: {scale_name}, root: {root_note} "
          f"({['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][root_note % 12]}"
          f"{root_note // 12 - 1})")
 
    # ── 1. Spine axes ────────────────────────────────────────────────────────
    spine_unit, perp_unit, bottom_arr = build_spine_axes(top, bottom)
    spine_length = float(np.linalg.norm(np.array(top) - np.array(bottom)))
 
    # ── 2. Spine nodes + BPM ────────────────────────────────────────────────
    spine_nodes = get_spine_nodes(G, top, bottom, spine_padding)
    if len(spine_nodes) < 2:
        raise ValueError(f"Only {len(spine_nodes)} spine nodes found. Increase spine_padding.")
 
    spine_nodes    = order_spine_nodes(spine_nodes, G, spine_unit, bottom_arr)
    spine_nodes_set = set(spine_nodes)
 
    raw_spine_lengths = [
        float(np.linalg.norm(get_node_position(G, spine_nodes[i]) -
                              get_node_position(G, spine_nodes[i + 1])))
        for i in range(len(spine_nodes) - 1)
    ]
    bpm = normalize_bpm(sum(raw_spine_lengths))
    drum_durations = normalize_durations(raw_spine_lengths, beats_per_bar)
 
    print(f"[green_music] BPM: {bpm}")
    print(f"[green_music] Spine nodes: {len(spine_nodes)}, drum durations: {drum_durations}")
 
    # ── 3. Quadrant nodes ───────────────────────────────────────────────────
    quadrants = get_quadrant_nodes(
        G, spine_nodes_set, bottom_arr, spine_unit, perp_unit, spine_length
    )
    for q, ns in quadrants.items():
        print(f"[green_music] Quadrant {q}: {len(ns)} nodes")
 
    # ── 4. Note lists per instrument ────────────────────────────────────────
    # Piano: 2 octaves from root (mid range)
    piano_notes   = build_note_list(root_note, scale_intervals, octaves=2)
    # Bass: 2 octaves one octave lower
    bass_notes    = build_note_list(root_note - 36, scale_intervals, octaves=2)
    bass_notes = [n for n in bass_notes if 24 <= n <= 48]  # C1–C2
    # Pad: same as piano range
    pad_notes     = build_note_list(root_note-12, scale_intervals, octaves=2)
 
    # ── 5. Generate melodies ────────────────────────────────────────────────
    # Piano uses both top quadrants merged
    piano_nodes  = quadrants["top_left"] + quadrants["top_right"]
    bass_nodes   = quadrants["bottom_left"]
    pad_nodes    = quadrants["bottom_right"]
 
    piano_events  = generate_melody_for_quadrant(
        G, piano_nodes,  piano_notes,  bottom_arr, spine_unit, perp_unit)
    bass_events   = generate_melody_for_quadrant(
        G, bass_nodes,   bass_notes,   bottom_arr, spine_unit, perp_unit)
    pad_events    = generate_melody_for_quadrant(
        G, pad_nodes,    pad_notes,    bottom_arr, spine_unit, perp_unit)
 
    print(f"[green_music] Piano events: {len(piano_events)}, "
          f"Bass: {len(bass_events)}, Pad: {len(pad_events)}")
 
    # ── 6. Total melody length (for drum loop) ──────────────────────────────
    def total_beats(events):
        return sum(d for _, d in events)
 
    melody_beats = max(
        total_beats(piano_events),
        total_beats(bass_events),
        total_beats(pad_events),
        beats_per_bar,
    )
    # Round up to nearest full bar
    total_bars    = math.ceil(melody_beats / beats_per_bar)
    total_beats_  = total_bars * beats_per_bar
 
    print(f"[green_music] Total bars: {total_bars}, total beats: {total_beats_}")
 
    # ── 7. Build MIDI ────────────────────────────────────────────────────────
    midi = MIDIFile(4)   # 5 tracks: drums, piano, bass, strings, pad
 
    for track in range(4):
        midi.addTempo(track, 0, bpm)
        midi.addTimeSignature(track, 0, beats_per_bar, 2, 24)
 
    # Program changes (instrument voices)
    midi.addProgramChange(TR_PIANO,   CH_PIANO,   0, GM_PIANO)
    midi.addProgramChange(TR_BASS,    CH_BASS,    0, GM_BASS)
    midi.addProgramChange(TR_PAD,     CH_PAD,     0, GM_PAD)
 
    SNARE_NOTE = 38

    # ── 7a. Drums — loop pattern for total_bars ──────────────────────────────
    for bar in range(total_bars):
        beat_time = bar * beats_per_bar
        drum_hit_index = 0
        for duration in drum_durations:
            if beat_time >= (bar + 1) * beats_per_bar:
                break
            note = KICK_NOTE if drum_hit_index % 2 == 0 else SNARE_NOTE
            midi.addNote(TR_DRUMS, CH_DRUMS, note,
                         time=beat_time, duration=0.25, volume=80)
            beat_time += duration
            drum_hit_index += 1
 
    # ── 7b. Helper to write event list to a track ───────────────────────────
    def write_events(track, channel, events, start_beat=0.0, vel=velocity):
        t = start_beat
        for pitch, duration in events:
            if pitch < 0 or pitch > 127:
                t += duration
                continue
            midi.addNote(track, channel, pitch,
                         time=t, duration=duration * 0.9,
                         volume=vel)
            t += duration

    # ── 7b2. Chord-aware writer for pad ─────────────────────────────────────
    def write_pad_events(track, channel, events, start_beat=0.0,
                         vel=70, chord_threshold=0.5):
        t = start_beat
        for pitch, duration in events:
            if pitch < 0 or pitch > 127:
                t += duration
                continue
            if duration > chord_threshold:
                # Build triad voiced below melody note and write all 3 notes
                chord_notes = build_chord(pitch, root_note, scale_name)
                for note in chord_notes:
                    midi.addNote(track, channel, note,
                                 time=t, duration=duration * 0.95,
                                 volume=vel)
            else:
                # Short note — just the melody note, no chord
                midi.addNote(track, channel, pitch,
                             time=t, duration=duration * 0.95,
                             volume=vel)
            t += duration

    # ── 7c. Piano (top quadrants) ────────────────────────────────────────────
    write_events(TR_PIANO, CH_PIANO, piano_events, vel=100)

    # ── 7d. Bass (bottom-left) ───────────────────────────────────────────────
    write_events(TR_BASS, CH_BASS, bass_events, vel=80)

    # ── 7e. Pad (bottom-right) — chords on longer notes ──────────────────────
    write_pad_events(TR_PAD, CH_PAD, pad_events, vel=50, chord_threshold=0.5)
 
    # ── 8. Write file ────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        midi.writeFile(f)
 
    print(f"[green_music] Written to: {output_path}")
 
    return {
        "bpm":          bpm,
        "scale":        scale_name,
        "root_note":    root_note,
        "total_bars":   total_bars,
        "spine_nodes":  len(spine_nodes),
        "piano_events": len(piano_events),
        "bass_events":  len(bass_events),
        "pad_events":   len(pad_events),
        "output_path":  output_path,
    }



# ─── Example usage ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from extractLeafVein import find_main_vein_endpoints, build_graph,extract_veins
    import cv2
    import os

    image_fileName = 'leaf2.png'
    image_path = f'leavesImages/{image_fileName}'

    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        exit(1)

    leafImage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(leafImage)

    vein = extract_veins(leafImage, enhanced)
    G = build_graph(leafImage, vein)

 

    n = list(G.nodes)[0]
    print("hey")
    print(G.nodes[n])
    print(type(G.nodes[n]['o']))
    print(G.nodes[n]['o'])
    # --- Replace with your actual getTopBottom call ---
    top, bottom = find_main_vein_endpoints(G,leafImage)

    generate_leaf_midi(G, top, bottom, output_path="leaf_drums.mid")