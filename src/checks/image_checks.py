from exceptions.pipeline_exceptions import ClutterCheckError, LowTextImageContrastRatioError
from settings.config import RetryConfig
from utils.image_utils import calculate_zone_edge_density, dominant_color_in_zone, contrast_ratio, hex_to_rgb_bytes
from PIL import Image

WCAG_AA_NORMAL = 4.5
WCAG_AA_LARGE  = 3.0

def check_zone_clutter(bg, text_design, cfg: RetryConfig = RetryConfig()):
    clutterness = calculate_zone_edge_density(bg, text_design)
    if clutterness["edge_density"] > cfg.clutter_threshold:
        raise ClutterCheckError(clutterness["edge_density"], cfg.clutter_threshold)
    return clutterness

def check_text_contrast(
    image: Image.Image,
    text_design,
    threshold: float = WCAG_AA_LARGE,
):
    bg_color = dominant_color_in_zone(image, text_design.text_zone)

    for role, text_color in [
        ("primary",   text_design.primary_color),
        ("secondary", text_design.secondary_color),
    ]:
        text_color = hex_to_rgb_bytes(text_color)
        ratio = contrast_ratio(bg_color, text_color)
        if ratio < threshold:
            raise LowTextImageContrastRatioError(
                bg_color=bg_color,
                text_color=text_color,
                ratio=ratio,
                role=role,
                threshold=threshold,
            )
