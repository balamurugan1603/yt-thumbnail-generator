from PIL import Image, ImageDraw, ImageFont
from core.models import TextDesign


def apply_directional_gradient(image: Image.Image, text_zone: str, darkness: int = 210) -> Image.Image:
    """Darken only the text zone with a directional gradient, leaving the subject vivid."""
    W, H   = image.size
    image  = image.convert("RGBA")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pixels  = overlay.load()
    zone    = text_zone.lower()

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
            alpha    = int(darkness * (y / band_h))
            for x in range(W):
                pixels[x, actual_y] = (0, 0, 0, alpha)

    elif zone == "center":
        cx, cy   = W // 2, H // 2
        max_dist = (cx ** 2 + cy ** 2) ** 0.5
        for x in range(W):
            for y in range(H):
                dist  = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                alpha = int(darkness * (dist / max_dist))
                pixels[x, y] = (0, 0, 0, alpha)

    return Image.alpha_composite(image, overlay).convert("RGB")


def _get_safe_zone(W: int, H: int, zone: str):
    """Return (safe_left, safe_right, safe_top, safe_bottom) for a text zone."""
    px, py = int(W * 0.04), int(H * 0.06)
    zones = {
        "left":        (px,            int(W * 0.50) - px, py,            H - py),
        "top-band":    (px,            W - px,             py,            int(H * 0.45) - py),
        "bottom-band": (px,            W - px,             int(H * 0.58) + py, H - py),
        "center":      (int(W * 0.10), W - int(W * 0.10),  int(H * 0.20), H - int(H * 0.20)),
    }
    return zones.get(zone, zones["center"])


def render_text(image: Image.Image, title: str, text_design: TextDesign) -> Image.Image:
    """Render title text centered (horizontally + vertically) inside its text zone."""
    draw = ImageDraw.Draw(image)
    W, H = image.size

    zone = text_design.text_zone.lower()
    safe_left, safe_right, safe_top, safe_bottom = _get_safe_zone(W, H, zone)
    safe_width  = safe_right  - safe_left
    safe_height = safe_bottom - safe_top

    emphasis_words = [w.upper() for w in text_design.emphasis_words]

    def _get_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype("arialbd.ttf", size)
        except:
            return ImageFont.load_default()

    # --- Find largest font that fits ---
    base_size    = 140
    min_size     = 36
    word_spacing = 20
    lines        = []
    line_heights = []
    total_height = 0

    while base_size > min_size:
        base_font = _get_font(base_size)
        lines, current = [], ""

        for word in title.upper().split():
            test = (current + " " + word).strip()
            if draw.textbbox((0, 0), test, font=base_font)[2] <= safe_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        line_heights, total_height = [], 0
        for line in lines:
            max_h = max(
                draw.textbbox((0, 0), w, font=_get_font(
                    int(base_size * text_design.emphasis_scale) if w in emphasis_words else base_size
                ))[3]
                for w in line.split()
            )
            line_heights.append(max_h)
            total_height += max_h + 20

        if total_height <= safe_height:
            break
        base_size -= 5

    # Vertically center the whole text block within the zone
    y = safe_top + (safe_height - total_height) // 2

    for i, line in enumerate(lines):
        line_height = line_heights[i]
        word_list   = line.split()

        # Pre-compute per-word font, width, height
        word_data = []
        for word in word_list:
            is_em = word in emphasis_words
            fs    = int(base_size * text_design.emphasis_scale) if is_em else base_size
            font  = _get_font(fs)
            bbox  = draw.textbbox((0, 0), word, font=font)
            word_data.append({
                "word":  word,
                "font":  font,
                "w":     bbox[2] - bbox[0],
                "h":     bbox[3] - bbox[1],
                "is_em": is_em,
            })

        true_line_w = sum(d["w"] for d in word_data) + word_spacing * (len(word_data) - 1)
        x_cursor    = safe_left + (safe_width - true_line_w) // 2

        for d in word_data:
            y_offset   = (line_height - d["h"]) // 2
            fill_color = text_design.secondary_color if d["is_em"] else text_design.primary_color

            # Shadow
            draw.text(
                (x_cursor + text_design.shadow_offset[0], y + y_offset + text_design.shadow_offset[1]),
                d["word"], font=d["font"], fill=text_design.shadow_color,
            )
            # Stroke + text
            draw.text(
                (x_cursor, y + y_offset),
                d["word"], font=d["font"], fill=fill_color,
                stroke_width=text_design.stroke_width, stroke_fill=text_design.stroke_color,
            )
            x_cursor += d["w"] + word_spacing

        y += line_height + 20

    return image