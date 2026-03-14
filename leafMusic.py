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


image_fileName = 'leaf4.png'  # Replace with your leaf image filename
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


# -----------------------------
# 3. Detect veins (Frangi)
# -----------------------------
from skimage import img_as_float

img_float = img_as_float(enhanced)

# Auto-detect vein polarity from leaf pixels only
_, rough_mask = cv2.threshold(leafImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# If mask covers >80% of image, leaf is dark on light background
if np.sum(rough_mask > 0) > 0.8 * leafImage.size:
    rough_mask = cv2.bitwise_not(rough_mask)
mean_val = img_float[rough_mask > 0].mean() if np.sum(rough_mask > 0) > 0 else img_float.mean()
black_ridges = mean_val > 0.5
print(f"Mean leaf brightness: {mean_val:.2f} -> black_ridges={black_ridges}")

vein_response = frangi(img_float, sigmas=range(1, 4, 1), black_ridges=black_ridges)
vein_norm = vein_response / (vein_response.max() + 1e-10)

# Threshold on NON-ZERO pixels only (Frangi is extremely skewed)
nonzero_vals = vein_norm[vein_norm > 0]
print(f"Non-zero pixels: {len(nonzero_vals)} ({100*len(nonzero_vals)/vein_norm.size:.1f}% of image)")

# Use percentile of non-zero values instead of Otsu
# Top X% of non-zero responses = veins
top_percent = 10  # keep top 10% of vein responses
auto_thresh = np.percentile(nonzero_vals, 100 - top_percent)
print(f"Auto threshold (top {top_percent}% of nonzero): {auto_thresh:.3f}")

vein_binary = (vein_norm > auto_thresh).astype(np.uint8) * 255

cv2.imshow("Frangi Veins", vein_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


# -----------------------------
# 4. Clean edges (morphology)
# -----------------------------
def estimate_morph_params(vein_binary):
    """Estimate close kernel size and iterations from gap sizes in the image."""
    
    # Find all white blobs (vein segments)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        vein_binary, connectivity=8
    )
    
    # Remove background (label 0)
    centroids = centroids[1:]
    stats = stats[1:]
    
    if len(centroids) < 2:
        return 5, 2  # fallback defaults
    
    # Measure distances between nearest blob pairs
    from scipy.spatial import KDTree
    tree = KDTree(centroids)
    # For each blob, find its nearest neighbor distance
    dists, _ = tree.query(centroids, k=2)  # k=2: itself + nearest
    nearest_dists = dists[:, 1]  # exclude self (distance=0)
    
    median_gap = np.median(nearest_dists)
    p75_gap    = np.percentile(nearest_dists, 75)
    
    print(f"Median gap between blobs: {median_gap:.1f}px")
    print(f"75th percentile gap:      {p75_gap:.1f}px")
    
    # Kernel should be ~half the gap size (morphology works from both sides)
    kernel_size = max(3, int(p75_gap / 2))
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd
    
    # Iterations: larger gaps need more passes
    iterations = max(1, int(median_gap / kernel_size) + 1)
    iterations = min(iterations, 6)  # cap at 6 to avoid over-merging
    
    print(f"Inferred kernel_size: {kernel_size}, iterations: {iterations}")
    return kernel_size, iterations


# Use it:
k, iters = estimate_morph_params(vein_binary)

kernel_close = np.ones((k, k), np.uint8)
kernel_open  = np.ones((3, 3), np.uint8)

clean = cv2.morphologyEx(vein_binary, cv2.MORPH_CLOSE, kernel_close, iterations=iters)  # use inferred iters, not hardcoded 1!
clean = cv2.morphologyEx(clean,       cv2.MORPH_OPEN,  kernel_open,  iterations=1)

cv2.imshow("Clean after morphology (before mask)", clean)  # check HERE before masking
cv2.waitKey(0)
cv2.destroyAllWindows()


# -----------------------------
# 4b. Remove leaf boundary
# -----------------------------
_, leaf_thresh = cv2.threshold(leafImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find all contours and pick largest
contours, _ = cv2.findContours(leaf_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_inv, _ = cv2.findContours(cv2.bitwise_not(leaf_thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Pick whichever threshold gives a contour that's NOT the whole image border
best_contour = None
best_area = 0
image_area = leafImage.shape[0] * leafImage.shape[1]

for c in list(contours) + list(contours_inv):
    area = cv2.contourArea(c)
    # Valid leaf contour: >5% and <95% of image area
    if 0.05 * image_area < area < 0.95 * image_area:
        if area > best_area:
            best_area = area
            best_contour = c

if best_contour is None:
    print("WARNING: Could not find leaf contour - skipping mask")
    clean_masked = clean
else:
    mask = np.zeros_like(clean)
    cv2.drawContours(mask, [best_contour], -1, 255, thickness=-1)
    mask = cv2.erode(mask, np.ones((7,7), np.uint8))
    print(f"Leaf mask area: {best_area} px ({100*best_area/image_area:.1f}% of image)")
    cv2.imshow("Leaf Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    clean = cv2.bitwise_and(clean, mask)

cv2.imshow("Clean before skeleton", clean)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------------
# 5. Skeletonize
# -----------------------------
skeleton = skeletonize(clean > 0).astype(np.uint8)
cv2.imshow("Skeletonized Veins", skeleton * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# 6. Build graph
# ----------------------------

G = sknw.build_sknw(skeleton > 0)
print(f"Raw: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---------------------------------------------------------------------
# 7. Clean graph
# -----------------------------

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