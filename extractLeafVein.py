import os
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import sknw
from isolate_leaf import isolate_leaf


def get_boundary_and_mask(image):
    """Returns both boundary points AND filled leaf mask using GrabCut."""
    if isinstance(image, str):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # GrabCut fallback for complex backgrounds
    print("Using GrabCut")
    mask = np.zeros(gray.shape[:2], np.uint8)
    h, w = gray.shape
    margin = int(min(h, w) * 0.1)
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("WARNING: GrabCut failed - using full image")
        return np.array([]), np.ones_like(gray) * 255

    largest = max(contours, key=cv2.contourArea)
    boundary_img = np.zeros_like(gray)
    cv2.drawContours(boundary_img, [largest], -1, 255, thickness=3)
    leaf_mask = np.zeros_like(gray)
    cv2.drawContours(leaf_mask, [largest], -1, 255, thickness=-1)

    pts = np.column_stack(np.where(boundary_img > 0))
    print(f"GrabCut boundary: {len(pts)} points")
    return pts, leaf_mask


def extract_veins(leafImage, enhanced):
    """Extract vein edges from leaf image using Canny + boundary barrier."""

    # Step 1: Canny edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges_canny = cv2.Canny(blurred, 50, 150)

    # Step 2: Leaf boundary + mask
    boundary_pts, leaf_mask = get_boundary_and_mask(leafImage)

    # Step 3: Build boundary canvas
    canvas_boundary = np.zeros(edges_canny.shape[:2], dtype=np.uint8)
    for pt in boundary_pts:
        r, c = int(pt[0]), int(pt[1])
        if 0 <= r < canvas_boundary.shape[0] and 0 <= c < canvas_boundary.shape[1]:
            canvas_boundary[r, c] = 255
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canvas_boundary = cv2.dilate(canvas_boundary, kernel_boundary)

    # Step 4: Veins = canny inside leaf mask, minus boundary
    vein = cv2.bitwise_and(edges_canny, cv2.bitwise_not(canvas_boundary))
    vein = cv2.bitwise_and(vein, leaf_mask)

    return vein


def connect_until_single_component(G):
    """Bridge components by increasing distance until fully connected (Kruskal-style)."""
    G = G.copy()

    all_nodes = list(G.nodes())
    if len(all_nodes) < 2:
        return G

    positions = np.array([G.nodes[n]['o'] for n in all_nodes])
    if positions.ndim != 2 or positions.shape[1] != 2:
        return G

    tree = KDTree(positions)
    dists, idxs = tree.query(positions, k=min(20, len(positions)))

    bridges = []
    for i, (d_row, idx_row) in enumerate(zip(dists, idxs)):
        u = all_nodes[i]
        for dist, j in zip(d_row, idx_row):
            v = all_nodes[j]
            if u != v:
                bridges.append((dist, u, v))

    bridges.sort(key=lambda x: x[0])

    added = 0
    last_dist = 0
    for dist, u, v in bridges:
        if nx.number_connected_components(G) == 1:
            break
        if not nx.has_path(G, u, v):
            pts = np.array([G.nodes[u]['o'], G.nodes[v]['o']])
            G.add_edge(u, v, pts=pts, weight=dist)
            added += 1
            last_dist = dist

    print(f"Added {added} bridges, max distance: {last_dist:.1f}px")
    print(f"Final components: {nx.number_connected_components(G)}")
    return G


def build_graph(leafImage, vein, visualize=True, save_path='graph_overlay.png'):
    """Build and clean graph from vein skeleton."""

    # Skeletonize
    skeleton = skeletonize(vein > 0).astype(np.uint8)

    # Build graph
    G = sknw.build_sknw(skeleton > 0)
    print(f"Raw: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Prune short edges scaled to image size
    img_diagonal = np.sqrt(leafImage.shape[0]**2 + leafImage.shape[1]**2)
    min_length = int(img_diagonal * 0.015)
    print(f"min_length: {min_length}")

    short = [(s, e) for s, e in G.edges() if len(G[s][e]['pts']) < min_length]
    G.remove_edges_from(short)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"After pruning: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Components: {nx.number_connected_components(G)}")

    # Connect components
    G = connect_until_single_component(G)

    if visualize:
        degrees = dict(G.degree())
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(leafImage, cmap='gray')
        for s, e in G.edges():
            pts = G[s][e]['pts']
            ax.plot(pts[:, 1], pts[:, 0], 'lime', linewidth=1.0, alpha=0.9)
        for n in G.nodes():
            pos = G.nodes[n]['o']
            color = 'red' if degrees[n] >= 3 else 'cyan'
            size  = 40   if degrees[n] >= 3 else 15
            ax.scatter(pos[1], pos[0], c=color, s=size, zorder=5)
        ax.set_title(f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges — red=junction cyan=tip')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Graph saved to {save_path}")
        plt.show()

    return G

# -----------------------------
# 9. Find main vein top & bottom
# -----------------------------
def find_main_vein_endpoints(G, leafImage):
    """
    Find the top and bottom of the main vein by locating the two degree-1
    endpoints (tips) with the greatest Euclidean distance between them.
    Returns (top_pt, bottom_pt) as (row, col) arrays.
    """
    tips = [n for n, d in G.degree() if d == 1]

    if len(tips) < 2:
        print("WARNING: fewer than 2 tip nodes found — falling back to all nodes")
        tips = list(G.nodes())

    tip_positions = np.array([G.nodes[n]['o'] for n in tips])  # (row, col)

    # ── Find the pair with maximum Euclidean distance ─────────────────────────
    tree = KDTree(tip_positions)
    # Query every tip against every other tip
    dists, idxs = tree.query(tip_positions, k=len(tip_positions))

    max_dist = -1
    top_pt, bottom_pt = None, None
    for i in range(len(tip_positions)):
        j = idxs[i, -1]                          # furthest neighbour of tip i
        d = dists[i, -1]
        if d > max_dist:
            max_dist = d
            top_pt   = tip_positions[i]
            bottom_pt = tip_positions[j]

    print(f"Main vein endpoint A : row={top_pt[0]:.0f},  col={top_pt[1]:.0f}")
    print(f"Main vein endpoint B : row={bottom_pt[0]:.0f}, col={bottom_pt[1]:.0f}")
    print(f"Euclidean distance   : {max_dist:.1f} px")

    # ── Visualise ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(leafImage, cmap='gray')

    for s, e in G.edges():
        pts = G[s][e]['pts']
        ax.plot(pts[:, 1], pts[:, 0], 'lime', linewidth=1.0, alpha=0.7)

    ax.scatter(tip_positions[:, 1], tip_positions[:, 0],
               c='cyan', s=15, zorder=4, label='Tips')
    ax.scatter(top_pt[1], top_pt[0],
               c='yellow', s=200, zorder=6, marker='*', label='Endpoint A')
    ax.scatter(bottom_pt[1], bottom_pt[0],
               c='red', s=200, zorder=6, marker='*', label='Endpoint B')
    ax.plot([top_pt[1], bottom_pt[1]], [top_pt[0], bottom_pt[0]],
            'white', linewidth=1.5, linestyle='--', alpha=0.8, label='Main axis')

    ax.legend(loc='upper right')
    ax.set_title(f'Main vein endpoints — furthest pair ({max_dist:.0f} px apart)')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('main_vein_endpoints.png', dpi=150)
    plt.show()
    
    return top_pt, bottom_pt

if __name__ == "__main__":
    image_fileName = 'leaf1.png'
    image_path = f'leavesImages/{image_fileName}'

    if not os.path.exists(image_path):
        print(f"Error: '{image_path}' not found.")
        exit(1)

    leafImage = cv2.imread(isolate_leaf(image_path), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(leafImage)

    vein = extract_veins(leafImage, enhanced)
    G = build_graph(leafImage, vein)

    print(f"Final graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")