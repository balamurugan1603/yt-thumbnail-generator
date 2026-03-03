import numpy as np
import cv2
from PIL import Image

def clutter_score_edge_density(pil_image):
    # Convert PIL to OpenCV format
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        "./assets/clutter_score_edge_density_RGB2BGR.jpg",
        img
    )

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    cv2.imwrite(
        "./assets/clutter_score_edge_density_edges.jpg",
        edges
    )

    # Calculate edge density
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size

    edge_density = edge_pixels / total_pixels
    return edge_density

def clutter_score_shannon_entropy(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()

    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    return entropy

def clutter_score_laplacian(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return variance

def check_zone_clutter(image: Image.Image, text_design) -> dict:
    """
    Returns clutter metrics for the LLM-selected text zone.
    """

    # --- Convert to OpenCV ---
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    H, W = gray.shape[:2]

    # --- Same zones as render_text ---
    zones = {
        "left":        (0.08, 0.50, 0.08, 0.92),
        "right":       (0.50, 0.92, 0.08, 0.92),
        "top-band":    (0.08, 0.92, 0.08, 0.50),
        "bottom-band": (0.08, 0.92, 0.50, 0.92),
        "center":      (0.08, 0.92, 0.08, 0.92),
    }

    zone = text_design["text_zone"].lower().strip()
    lx, rx, ty, by = zones.get(zone, zones["center"])

    SL, SR = int(W * lx), int(W * rx)
    ST, SB = int(H * ty), int(H * by)

    region = gray[ST:SB, SL:SR]

    # --- Resize for resolution normalization ---
    # target_width = 512
    # scale = target_width / region.shape[1]
    # region = cv2.resize(region, (target_width, int(region.shape[0] * scale)))

    # --- Edge Density ---
    edges = cv2.Canny(region, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return {
        "zone": zone,
        "edge_density": float(edge_density)
    }

path = "../src/artifacts/bg-1772517162.877577.png"
img = Image.open(path)
print("CLUTTERNESS SCORE: ", clutter_score_edge_density(img))
print("CLUTTERNESS SCORE SHANNON ENTROPY: ", clutter_score_shannon_entropy(img))
print("CLUTTERNESS SCORE LAPLACIAN: ", clutter_score_laplacian(img))
print("ZONE CLUTTER: ", check_zone_clutter(img, text_design={'text_zone': 'left'}))