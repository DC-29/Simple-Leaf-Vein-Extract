import networkx as nx
import numpy as np
from midiutil import MIDIFile
 
 
# ─── Helpers ──────────────────────────────────────────────────────────────────
 
def get_node_position(G: nx.Graph, node: int) -> np.ndarray:
    """Return [y, x] position of a node."""
    return np.array(G.nodes[node]['o'])
 
 
def find_nearest_node(G: nx.Graph, coord: list) -> int:
    """Return the node ID whose position is closest to coord ([y, x])."""
    coord = np.array(coord)
    return min(G.nodes, key=lambda n: np.linalg.norm(get_node_position(G, n) - coord))
 
 
 
 
def normalize_durations(raw_lengths: list, beats_per_bar: int = 16, min_step: float = 0.125) -> list:
    total = sum(raw_lengths)
    if total == 0:
        n = max(len(raw_lengths), 1)
        return [min_step] * n

    # Square root to exaggerate relative differences
    # e.g. 65k → 255, 92k → 303  (ratio 1.19x instead of 1.42x)
    # You can swap np.sqrt for np.log to exaggerate even more aggressively
    transformed = [l**0.1 for l in raw_lengths]

    # Now normalize transformed values to sum to beats_per_bar
    t_total = sum(transformed)
    scale = beats_per_bar / t_total
    scaled = [l * scale for l in transformed]

    # Round to nearest min_step
    step_units = int(round(1 / min_step))  # e.g. 8 for 0.125
    rounded = [max(min_step, round(l * step_units) / step_units) for l in scaled]

    # Fix rounding drift so sum == beats_per_bar exactly
    target_steps = int(round(beats_per_bar / min_step))
    current_steps = int(round(sum(rounded) / min_step))
    diff = target_steps - current_steps

    indices = sorted(range(len(rounded)), key=lambda i: rounded[i], reverse=(diff < 0))
    for i in indices:
        if diff == 0:
            break
        if diff > 0:
            rounded[i] += min_step
            diff -= 1
        elif rounded[i] > min_step:
            rounded[i] -= min_step
            diff += 1

    return rounded
 
 
def normalize_bpm(spine_length: float,
                  min_length: float = 450.0,
                  max_length: float = 500.0,
                  min_bpm: int = 60,
                  max_bpm: int = 110) -> int:
    """
    Map spine length linearly to a BPM in [min_bpm, max_bpm].
    Longer spine → slower BPM.
    """
    clamped = max(min_length, min(max_length, spine_length))
    ratio = (clamped - min_length) / (max_length - min_length)
    bpm = max_bpm - ratio * (max_bpm - min_bpm)
    return int(round(bpm))
 
 
# ─── Spine detection ──────────────────────────────────────────────────────────
 
def get_spine_nodes(G: nx.Graph, top: list, bottom: list, padding: int = 10) -> list:
    """
    Return all node IDs whose [y, x] position lies within the bounding
    rectangle formed by top and bottom, with ±padding on the x axis.
    """
    x_min = min(top[1], bottom[1]) - padding
    x_max = max(top[1], bottom[1]) + padding
    y_min = min(top[0], bottom[0])
    y_max = max(top[0], bottom[0])


    # Check how many nodes fall in range
    count = 0
    degrees = dict(G.degree())
    spine_nodes = []
    for n in G.nodes:
        pos = G.nodes[n]['o']

        ny, nx_ = pos[0], pos[1]
        if x_min <= nx_ <= x_max and y_min <= ny <= y_max and degrees[n]>3:
            count += 1
            spine_nodes.append(n)
            print(f"  node {n}: y={ny}, x={nx_}")
 
    return spine_nodes
 
 
def order_spine_nodes(spine_nodes: list, G: nx.Graph, bottom: list) -> list:
    """
    Order spine nodes from bottom to top by projecting onto the spine vector.
    """
    bottom_arr = np.array(bottom)
    positions = [get_node_position(G, n) for n in spine_nodes]
 
    # Spine direction: from bottom toward the node with smallest y (top of image)
    top_pos = min(positions, key=lambda p: p[0])
    spine_vec = top_pos - bottom_arr
    spine_len = np.linalg.norm(spine_vec)
    if spine_len == 0:
        return spine_nodes
 
    unit = spine_vec / spine_len
 
    def projection(n):
        pos = get_node_position(G, n)
        return float(np.dot(pos - bottom_arr, unit))
 
    return sorted(spine_nodes, key=projection)
 
 
# ─── MIDI generation ──────────────────────────────────────────────────────────
 
KICK_NOTE    = 36   # Bass drum (General MIDI)
DRUM_CHANNEL = 9    # GM drum channel (0-indexed)
DRUM_TRACK   = 0
 
 
def generate_drum_midi(
    G: nx.Graph,
    top: list,
    bottom: list,
    output_path: str = "leaf_drums.mid",
    beats_per_bar: int = 4,
    velocity: int = 100,
    padding: int = 10,
) -> dict:
    """
    Generate a single-bar drum MIDI file from a leaf skeleton graph.
 
    The mid-ridge nodes define kick drum hit positions within exactly
    one bar. BPM is derived from the total spine length.
 
    Parameters
    ----------
    G             : NetworkX graph from sknw.build_sknw
    top           : [y, x] top of the mid-ridge
    bottom        : [y, x] bottom of the mid-ridge
    output_path   : path to write the .mid file
    beats_per_bar : time signature numerator (4 for 4/4)
    velocity      : MIDI velocity for kick hits (0–127)
    padding       : ±x pixel width of the spine bounding box
 
    Returns
    -------
    dict with bpm, spine_node_count, durations, output_path
    """
 
    # 1. Find and order spine nodes
    spine_nodes = get_spine_nodes(G, top, bottom, padding)
    if len(spine_nodes) < 2:
        raise ValueError(
            f"Only {len(spine_nodes)} spine node(s) found. "
            "Try increasing padding or check top/bottom coordinates."
        )
    spine_nodes = order_spine_nodes(spine_nodes, G, bottom)
 
    # 2. Euclidean distances between consecutive spine nodes
    raw_lengths = []
    for i in range(len(spine_nodes) - 1):
        p1 = get_node_position(G, spine_nodes[i])
        p2 = get_node_position(G, spine_nodes[i + 1])
        raw_lengths.append(float(np.linalg.norm(p1 - p2)))
 
    # 3. BPM from total spine length
    spine_length = sum(raw_lengths)
    bpm = normalize_bpm(spine_length)
    print(raw_lengths)
    # 4. Normalize durations — guaranteed to sum to exactly beats_per_bar
    durations = normalize_durations(raw_lengths, beats_per_bar)
    print(durations)
    assert abs(sum(durations) - beats_per_bar) < 1e-9, \
        f"Duration sum {sum(durations)} != {beats_per_bar}"
 
    # 5. Build MIDI — exactly 1 bar
    midi = MIDIFile(1)
    midi.addTempo(DRUM_TRACK, 0, bpm)
    # midi.addTimeSignature(
    #     track=DRUM_TRACK,
    #     time=0,
    #     numerator=beats_per_bar,
    #     denominator=4,          
    #     clocks_per_tick=24,
    # )
 
    # Place kicks at cumulative beat times — all within [0, beats_per_bar)
    beat_time = 0.0
    for d in durations:
        # Sanity guard: don't write past the bar
        if beat_time >= beats_per_bar:
            break
        midi.addNote(
            track=DRUM_TRACK,
            channel=DRUM_CHANNEL,
            pitch=KICK_NOTE,
            time=beat_time,
            duration=0.25,      # kick hit is always a short transient
            volume=velocity,
        )
        beat_time += d
 
    # 6. Write file
    with open(output_path, "wb") as f:
        midi.writeFile(f)
 
    return {
        "bpm": bpm,
        "spine_node_count": len(spine_nodes),
        "raw_lengths": raw_lengths,
        "durations": durations,
        "output_path": output_path,
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

    generate_drum_midi(G, top, bottom, output_path="leaf_drums.mid")


