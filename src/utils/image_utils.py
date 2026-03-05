import re

from PIL import Image, ImageDraw, ImageFont
from core.models import TextDesign
import numpy as np
from numpy.typing import NDArray
import base64
from io import BytesIO
import cv2
import pytesseract
from settings.config import TESSERACT_PATH

ZONES = {
    "left": (0.08, 0.50, 0.08, 0.92),
    "right": (0.50, 0.92, 0.08, 0.92),
    "top-band": (0.08, 0.92, 0.08, 0.50),
    "bottom-band": (0.08, 0.92, 0.50, 0.92),
    "center": (0.08, 0.92, 0.08, 0.92),
}


def apply_directional_gradient(
    image: Image.Image, text_zone: str, darkness: int = 210
) -> Image.Image:
    """Darken only the text zone with a directional gradient, leaving the subject vivid."""
    W, H = image.size
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pixels = overlay.load()
    zone = text_zone.lower()

    if zone == "left":
        band_w = int(W * 0.50)
        for x in range(band_w):
            alpha = int(darkness * (1 - x / band_w))
            for y in range(H):
                pixels[x, y] = (0, 0, 0, alpha)

    elif zone == "top-band":
        band_h = int(H * 0.45)
        for y in range(band_h):
            alpha = int(darkness * (1 - y / band_h))
            for x in range(W):
                pixels[x, y] = (0, 0, 0, alpha)

    elif zone == "bottom-band":
        band_h = int(H * 0.42)
        for y in range(band_h):
            actual_y = H - band_h + y
            alpha = int(darkness * (y / band_h))
            for x in range(W):
                pixels[x, actual_y] = (0, 0, 0, alpha)

    elif zone == "center":
        cx, cy = W // 2, H // 2
        max_dist = (cx**2 + cy**2) ** 0.5
        for x in range(W):
            for y in range(H):
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                alpha = int(darkness * (dist / max_dist))
                pixels[x, y] = (0, 0, 0, alpha)

    return Image.alpha_composite(image, overlay).convert("RGB")


def render_text(image: Image.Image, title: str, text_design: TextDesign) -> Image.Image:
    import re

    GAP = 16
    LINE_SPACING = 14
    RENDER_SIZE = 120
    sw = text_design.stroke_width

    emphasis_set = {
        re.sub(r"[^A-Z0-9]", "", w.upper()) for w in text_design.emphasis_words
    }

    def is_emp(word: str) -> bool:
        return re.sub(r"[^A-Z0-9]", "", word.upper()) in emphasis_set

    def load_font(size: int):
        try:
            return ImageFont.truetype("arialbd.ttf", max(size, 1))
        except:
            return ImageFont.load_default()

    base_font = load_font(RENDER_SIZE)
    emp_font = load_font(int(RENDER_SIZE * text_design.emphasis_scale))

    # ── 1. Define safe zone first so we know wrap width ──────────────────
    W, H = image.size
    zone = text_design.text_zone.lower().strip()
    zones = {
        "left": (0.08, 0.50, 0.08, 0.92),
        "right": (0.50, 0.92, 0.08, 0.92),
        "top-band": (0.08, 0.92, 0.08, 0.50),
        "bottom-band": (0.08, 0.92, 0.50, 0.92),
        "center": (0.08, 0.92, 0.08, 0.92),
    }
    lx, rx, ty, by = zones.get(zone, zones["center"])
    SL, SR = int(W * lx), int(W * rx)
    ST, SB = int(H * ty), int(H * by)
    SAFE_W = SR - SL
    SAFE_H = SB - ST

    # ── 2. Word-wrap using safe wrap width ────────────────────────────
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def word_dim(word, font):
        bb = dummy.textbbox((0, 0), word, font=font, stroke_width=sw)
        return bb[2] - bb[0], bb[3] - bb[1]

    lines, current, current_w = [], [], 0
    for word in title.upper().split():
        f = emp_font if is_emp(word) else base_font
        ww, _ = word_dim(word, f)
        needed = ww if not current else ww + GAP
        if current and current_w + needed > SAFE_W:
            lines.append(current)
            current, current_w = [word], ww
        else:
            current.append(word)
            current_w += needed
    if current:
        lines.append(current)

    # ── 3. Measure full text block ────────────────────────────────────────
    line_dims = []
    for line in lines:
        lw = lh = 0
        for i, word in enumerate(line):
            f = emp_font if is_emp(word) else base_font
            ww, wh = word_dim(word, f)
            lw += ww + (GAP if i > 0 else 0)
            lh = max(lh, wh)
        line_dims.append((lw, lh))

    block_w = max(lw for lw, _ in line_dims)
    block_h = sum(lh for _, lh in line_dims) + LINE_SPACING * (len(lines) - 1)

    # Add padding so stroke + shadow don't get clipped at canvas edges
    PAD = (
        sw
        + max(abs(text_design.shadow_offset[0]), abs(text_design.shadow_offset[1]))
        + 4
    )
    canvas_w = block_w + PAD * 2
    canvas_h = block_h + PAD * 2

    # ── 4. Render text onto transparent canvas ───────────────────────────
    PAD = (
        sw
        + max(abs(text_design.shadow_offset[0]), abs(text_design.shadow_offset[1]))
        + 60
    )
    canvas = Image.new("RGBA", (block_w + PAD * 2, block_h + PAD * 2), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(canvas)

    y = PAD
    for (lw, lh), line in zip(line_dims, lines):
        x = PAD + (block_w - lw) // 2

        for i, word in enumerate(line):
            f = emp_font if is_emp(word) else base_font
            fill = (
                text_design.secondary_color
                if is_emp(word)
                else text_design.primary_color
            )
            ww, wh = word_dim(word, f)
            wy = y + (lh - wh) // 2

            cdraw.text(
                (x + text_design.shadow_offset[0], wy + text_design.shadow_offset[1]),
                word,
                font=f,
                fill=text_design.shadow_color,
            )
            cdraw.text(
                (x, wy),
                word,
                font=f,
                fill=fill,
                stroke_width=sw,
                stroke_fill=(
                    text_design.secondary_stroke_color
                    if is_emp(word)
                    else text_design.primary_stroke_color
                ),
            )
            x += ww + GAP

        y += lh + LINE_SPACING

    # ── 5. Crop to actual rendered pixels (eliminates all measurement error)
    bbox = (
        canvas.getbbox()
    )  # returns (left, top, right, bottom) of non-transparent pixels
    if bbox:
        canvas = canvas.crop(bbox)

    # ── 6. Scale to fit safe zone preserving aspect ratio ────────────────
    cw, ch = canvas.size
    scale = min(SAFE_W / cw, SAFE_H / ch)
    new_w = int(cw * scale)
    new_h = int(ch * scale)
    canvas = canvas.resize((new_w, new_h), Image.LANCZOS)

    # ── 7. Paste centered within safe zone ───────────────────────────────
    if zone == "top-band":
        paste_y = ST
    elif zone == "bottom-band":
        paste_y = SB - new_h
    else:
        paste_y = ST + (SAFE_H - new_h) // 2

    if zone == "left":
        paste_x = SL
    elif zone == "right":
        paste_x = SR - new_w
    else:
        paste_x = SL + (SAFE_W - new_w) // 2

    image = image.convert("RGBA")
    image.paste(canvas, (paste_x, paste_y), canvas)
    return image.convert("RGB")


def pil_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL image to a base64-encoded string."""
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def rgb_to_sobel(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Convert an RGB/BGR image to Sobel edge magnitude image.

    Args:
        image: Input image as uint8 NumPy array of shape (H, W, 3)

    Returns:
        Sobel magnitude image as uint8 NumPy array of shape (H, W, 3)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    sobel_bgr = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2BGR)
    sobel_rgb = cv2.cvtColor(sobel_bgr, cv2.COLOR_BGR2RGB)
    return sobel_rgb


def calculate_zone_edge_density(image: Image.Image, text_design: TextDesign) -> dict:
    """
    Returns clutter metrics for the LLM-selected text zone.
    """

    # --- Convert to OpenCV ---
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    H, W = gray.shape[:2]

    zone = text_design.text_zone.lower().strip()
    lx, rx, ty, by = ZONES.get(zone, ZONES["center"])

    SL, SR = int(W * lx), int(W * rx)
    ST, SB = int(H * ty), int(H * by)

    region = gray[ST:SB, SL:SR]

    # --- Edge Density ---
    edges = cv2.Canny(region, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return {"zone": zone, "edge_density": float(edge_density)}


def get_zone_bounds(image: Image.Image, text_zone: str) -> tuple[int, int, int, int]:
    """
    Resolves a zone name to pixel bounds (SL, SR, ST, SB),
    matching the same coordinate logic as calculate_zone_edge_density.
    """
    W, H = image.size
    lx, rx, ty, by = ZONES.get(text_zone.lower().strip(), ZONES["center"])
    SL, SR = int(W * lx), int(W * rx)
    ST, SB = int(H * ty), int(H * by)
    return SL, SR, ST, SB


def dominant_color_in_zone(image: Image.Image, text_zone: str) -> tuple[int, int, int]:
    """
    Crops to the named text zone and returns the dominant color
    after median-cut quantization to 8 buckets.
    """
    SL, SR, ST, SB = get_zone_bounds(image, text_zone)
    region = image.crop((SL, ST, SR, SB)).convert("RGB")

    quantized = region.quantize(colors=8, method=Image.Quantize.MEDIANCUT).convert(
        "RGB"
    )
    pixels = np.array(quantized).reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]
    return tuple(int(c) for c in dominant)


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    def channel(c: int) -> float:
        s = c / 255.0
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast_ratio(color_a: tuple, color_b: tuple) -> float:
    la, lb = relative_luminance(color_a), relative_luminance(color_b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def hex_to_rgb_bytes(hex_code):
    hex_code = hex_code.lstrip("#")
    return tuple(bytes.fromhex(hex_code))


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extracts all text from an image and formats it as a single sentence.
    """
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(image)
    clean_text = extracted_text.replace("\x0c", "").strip()
    single_line_text = re.sub(r"\s+", " ", clean_text)
    return single_line_text
