from exceptions.pipeline_exceptions import ClutterCheckError, LowTextImageContrastRatioError, ArtifactDetectedError
from settings.config import RetryConfig
from utils.image_utils import calculate_zone_edge_density, dominant_color_in_zone, contrast_ratio, hex_to_rgb_bytes
from services.clip_service import load_clip_zero_shot
from PIL import Image

WCAG_AA_NORMAL = 4.5
WCAG_AA_LARGE  = 3.0

CONTENT_PROMPT_PAIRS = [
    (
        "an image without face, hand, human artifacts, text, typography or any symbol",
        "an image with face, hand, human artifacts, text, typography or any symbol",
    ),
]

def check_background_content(image: Image.Image) -> dict:
    """
    Checks the background has no faces, hands, text or symbols.
    Returns per-pair scores for logging. Raises ArtifactDetectedError if any pair fails.
    """
    clip       = load_clip_zero_shot()
    all_labels = [label for pair in CONTENT_PROMPT_PAIRS for label in pair]
    results    = clip(image, candidate_labels=all_labels)
    scores     = {r["label"]: r["score"] for r in results}

    pair_results = []
    for pos, neg in CONTENT_PROMPT_PAIRS:
        pos_score = scores[pos]
        neg_score = scores[neg]
        passed    = pos_score > neg_score
        pair_results.append({
            "positive": pos, "pos_score": pos_score,
            "negative": neg, "neg_score": neg_score,
            "passed":   passed,
        })
        if not passed:
            raise ArtifactDetectedError(pos_score, neg_score)
    return pair_results

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

