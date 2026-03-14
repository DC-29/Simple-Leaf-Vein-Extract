import os
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import mido
import matplotlib.pyplot as plt
from skimage.filters import frangi
from scipy.spatial import KDTree
import sknw


print("hello, imports work!")

# -----------------------------
# 1. Load the leaf image
# -----------------------------


image_fileName = 'leaf8.png'  # Replace with your leaf image filename
image_path = f'SimpleLeafVeinExtraction/leavesImages/{image_fileName}'

if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    exit(1)

print("Leaf image loaded successfully!")
leafImage = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Display the image in a window
# cv2.imshow("Loaded Image", leafImage) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# -----------------------------
# 2. Enhance image constrast
# -----------------------------

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(leafImage)
# cv2.imshow("Enhanced Image", enhanced) # Display the enhanced image in a window
# cv2.waitKey(0) # Wait for a key press to close the window
# cv2.destroyAllWindows() 


def get_boundary(image):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    else:
        img = image.copy()
    
    # Use Otsu to separate leaf from background
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If leaf is dark on white background, invert
    if np.sum(binary > 0) > 0.5 * binary.size:
        binary = cv2.bitwise_not(binary)
    
    # Find outer contour only
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
    
    leaf_contour = max(contours, key=cv2.contourArea)
    
    # Draw just the outline
    boundary_img = np.zeros_like(img)
    cv2.drawContours(boundary_img, [leaf_contour], -1, 255, thickness=3)
    
    # Return as point array
    pts = np.column_stack(np.where(boundary_img > 0))
    print(f"Boundary points: {len(pts)}")
    return pts



def region_grow_from_top(vein_img):
    """Get largest connected component that contains the topmost significant pixel."""
    
    # Label all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        vein_img, connectivity=8
    )
    
    if num_labels <= 1:
        return np.zeros_like(vein_img)
    
    # Find largest component (skip background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    print(f"Largest component area: {stats[largest_label, cv2.CC_STAT_AREA]} px")
    print(f"Total components: {num_labels - 1}")
    
    main_vein = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return main_vein

def extract_veins(leafImage, enhanced):
    # Step 1: Canny edges
    blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
    edges_canny = cv2.Canny(blurred, 50, 150)
    
    cv2.imshow("Canny Edges", edges_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Step 2: Leaf boundary as barrier
    boundary_pts = get_boundary(leafImage)
    canvas_boundary = np.zeros(edges_canny.shape[:2], dtype=np.uint8)
    for pt in boundary_pts:
        r, c = int(pt[0]), int(pt[1])
        if 0 <= r < canvas_boundary.shape[0] and 0 <= c < canvas_boundary.shape[1]:
            canvas_boundary[r, c] = 255
    kernel_boundary = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    canvas_boundary = cv2.dilate(canvas_boundary, kernel_boundary)

    # Step 3: Veins = canny edges inside leaf, minus boundary
    _, leaf_binary = cv2.threshold(leafImage, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.sum(leaf_binary > 0) > 0.5 * leaf_binary.size:
        leaf_binary = cv2.bitwise_not(leaf_binary)

    vein = cv2.bitwise_and(edges_canny, cv2.bitwise_not(canvas_boundary))
    vein = cv2.bitwise_and(vein, leaf_binary)

    # show vein and print pixel count
    print(f"Vein white pixels: {np.sum(vein > 0)}")
    cv2.imshow("Veins raw", vein)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return vein, vein  # return vein for both


vein, main_vein = extract_veins(leafImage, enhanced)

# Skeletonize main vein
skeleton = skeletonize(main_vein > 0).astype(np.uint8)
cv2.imshow("Skeleton", skeleton * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# 6. Build graph
# ----------------------------

G = sknw.build_sknw(skeleton > 0)
raw_nodes = G.number_of_nodes()
print(f"Raw nodes: {raw_nodes}")


# ---------------------------------------------------------------------
# 7. Clean graph
# -----------------------------

# Aggressive small spurious branches removal scaled to image size
img_diagonal = np.sqrt(leafImage.shape[0]**2 + leafImage.shape[1]**2)
min_length = int(img_diagonal * 0.015)  # 2% of diagonal
print(f"min_length: {min_length}")

short = [(s, e) for s, e in G.edges() if len(G[s][e]['pts']) < min_length]
G.remove_edges_from(short)
G.remove_nodes_from(list(nx.isolates(G)))
print(f"After pruning: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Components: {nx.number_connected_components(G)}")

# Use Kruskal-like approach to connect components by adding edges between nearest nodes across components, 
# until we have a single connected component. 

def connect_until_single_component(G):
    """Bridge components by increasing distance until fully connected."""
    G = G.copy()
    
    all_nodes = list(G.nodes())
    positions = np.array([G.nodes[n]['o'] for n in all_nodes])
    tree = KDTree(positions)
    
    # Pre-compute all cross-component distances sorted ascending
    dists, idxs = tree.query(positions, k=min(20, len(positions)))
    
    # Collect all candidate bridges
    bridges = []
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    
    for i, (d_row, idx_row) in enumerate(zip(dists, idxs)):
        u = all_nodes[i]
        for dist, j in zip(d_row, idx_row):
            v = all_nodes[j]
            if u != v:
                bridges.append((dist, u, v))
    
    # Sort by distance - add shortest bridges first (Kruskal-style)
    bridges.sort(key=lambda x: x[0])
    
    added = 0
    for dist, u, v in bridges:
        if nx.number_connected_components(G) == 1:
            break
        # Only add if still in different components
        if not nx.has_path(G, u, v):
            pts = np.array([G.nodes[u]['o'], G.nodes[v]['o']])
            G.add_edge(u, v, pts=pts, weight=dist)
            added += 1
    
    print(f"Added {added} bridges")
    print(f"Max bridge distance used: {dist:.1f}px")
    print(f"Final: {nx.number_connected_components(G)} components")
    return G

G = connect_until_single_component(G)

print(f"Number of components: {nx.number_connected_components(G)}")



# -----------------------------
# 8. Visualize overlay on leaf
# -----------------------------

degrees = dict(G.degree())
nodes_arr = np.array([G.nodes[n]['o'] for n in G.nodes()])

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
plt.savefig('graph_overlay.png', dpi=150)
plt.show()
