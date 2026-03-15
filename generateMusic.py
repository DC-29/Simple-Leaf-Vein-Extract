import os
import cv2
import mido
import random
import numpy as np
from mido import MidiFile, MidiTrack, Message
from extractLeafVein import extract_veins, build_graph
import argparse


def graph_to_midi(G, leafImage, output_file='leaf_music.mid'):
    """Convert leaf vein graph to MIDI file."""

    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # Random tempo 60-140 BPM for experimental feel
    bpm = random.randint(60, 140)
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    print(f"BPM: {bpm}")

    # Pentatonic minor scale (eerie/organic feel)
    # Alternatives:
    # [0,2,4,7,9]       = pentatonic major (bright)
    # [0,2,3,5,7,8,10]  = natural minor (dark)
    # [0,1,2,3,4,5,6]   = chromatic (very experimental)
    scale_intervals = [0, 3, 5, 7, 10]
    root_note = random.randint(36, 60)
    print(f"Root note: {root_note}, Scale: pentatonic minor")

    def pitch_from_y(y, img_height):
        """Map y position to pitch — top=high, bottom=low."""
        y_norm = 1 - (y / img_height)
        octave = int(y_norm * 3)
        scale_degree = int(y_norm * len(scale_intervals) * 3) % len(scale_intervals)
        pitch = root_note + octave * 12 + scale_intervals[scale_degree]
        return max(0, min(127, pitch))

    def get_chord(root_pitch):
        """Build a 3-note chord from root pitch."""
        return [
            root_pitch,
            min(127, root_pitch + scale_intervals[1]),  # minor 3rd
            min(127, root_pitch + scale_intervals[2]),  # 5th
        ]

    h, w = leafImage.shape
    degrees = dict(G.degree())
    nodes_pos = {n: G.nodes[n]['o'] for n in G.nodes()}

    # BFS traversal from topmost node (leaf tip)
    start_node = min(G.nodes(), key=lambda n: nodes_pos[n][0])
    print(f"Starting BFS from node {start_node} at {nodes_pos[start_node]}")

    visited_edges = set()
    visited_nodes = {start_node}
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        pos = nodes_pos[node]
        is_junction = degrees[node] >= 3

        for neighbor in G.neighbors(node):
            edge_key = tuple(sorted([node, neighbor]))
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            pts = G[node][neighbor]['pts']
            pitch = pitch_from_y(pos[0], h)

            # Duration from edge length + random variation
            edge_length = len(pts)
            base_duration = max(100, min(800, edge_length * 8))
            duration = base_duration + random.randint(-50, 50)
            ticks = int(duration * mid.ticks_per_beat / 500)

            # Occasional random rests for experimental feel
            rest_ticks = random.choice([0, 0, 0, ticks // 4, ticks // 2])

            if is_junction:
                # Chord at junction nodes (branching points)
                chord = get_chord(pitch)
                velocity = random.randint(70, 100)
                for i, note in enumerate(chord):
                    track.append(Message('note_on', note=note,
                                         velocity=velocity,
                                         time=rest_ticks if i == 0 else 0))
                for i, note in enumerate(chord):
                    track.append(Message('note_off', note=note,
                                          velocity=0,
                                          time=ticks if i == 0 else 0))
            else:
                # Single note at tip/regular nodes
                velocity = random.randint(40, 70)
                track.append(Message('note_on',  note=pitch,
                                     velocity=velocity, time=rest_ticks))
                track.append(Message('note_off', note=pitch,
                                     velocity=0,        time=ticks))

            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append(neighbor)

    mid.save(output_file)
    print(f"Saved: {output_file}")
    print(f"Total notes: {len(visited_edges)}, BPM: {bpm}, Root: {root_note}")
    return mid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    args = parser.parse_args()
    image_fileName = args.image
    image_path = f'leavesImages/{image_fileName}'

    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        exit(1)

    leafImage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(leafImage)

    vein = extract_veins(leafImage, enhanced)
    G = build_graph(leafImage, vein, visualize=False)
    print(G)

    graph_to_midi(G, leafImage, output_file='leaf_music.mid')