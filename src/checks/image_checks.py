from exceptions.pipeline_exceptions import (
    ClutterCheckError,
    LowTextImageContrastRatioError,
    ArtifactDetectedError,
    PromptImageSimilarityError,
)
from settings.config import RetryConfig
from utils.image_utils import (
    calculate_zone_edge_density,
    dominant_color_in_zone,
    contrast_ratio,
    extract_text_from_image,
    hex_to_rgb_bytes,
)
from services.clip_service import (
    get_model_and_processor,
    load_clip_zero_shot,
    clip_score,
)
from PIL import Image
import Levenshtein

CONTENT_PROMPT_PAIRS = [
    (
        "an image without face, hand, human artifacts, text, typography or any symbol",
        "an image with face, hand, human artifacts, text, typography or any symbol",
    ),
]

clip_zero_shot_pipeline = load_clip_zero_shot()
clip_model, processor = get_model_and_processor(device="cpu")


def check_background_content(image: Image.Image) -> dict:
    """
    Checks the background has no faces, hands, text or symbols.
    Returns per-pair scores for logging. Raises ArtifactDetectedError if any pair fails.
    """
    all_labels = [label for pair in CONTENT_PROMPT_PAIRS for label in pair]
    results = clip_zero_shot_pipeline(image, candidate_labels=all_labels)
    scores = {r["label"]: r["score"] for r in results}

    pair_results = []
    for pos, neg in CONTENT_PROMPT_PAIRS:
        pos_score = scores[pos]
        neg_score = scores[neg]
        passed = pos_score > neg_score
        pair_results.append(
            {
                "positive": pos,
                "pos_score": pos_score,
                "negative": neg,
                "neg_score": neg_score,
                "passed": passed,
            }
        )
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
    cfg: RetryConfig = RetryConfig(),
):
    bg_color = dominant_color_in_zone(image, text_design.text_zone)

    ratios = {}
    for role, text_color in [
        ("primary", text_design.primary_color),
        ("secondary", text_design.secondary_color),
    ]:
        text_color = hex_to_rgb_bytes(text_color)
        ratio = contrast_ratio(bg_color, text_color)
        ratios[role] = ratio

    for role, ratio in ratios.items():
        if ratio < cfg.contrast_ratio_threshold:
            raise LowTextImageContrastRatioError(
                bg_color=bg_color,
                primary_text_color=text_design.primary_color,
                secondary_text_color=text_design.secondary_color,
                primary_contrast_ratio=ratios.get("primary"),
                secondary_contrast_ratio=ratios.get("secondary"),
                role=role,
                threshold=cfg.contrast_ratio_threshold,
            )
    return ratios


def check_clip_score(image: Image.Image, prompt: str, cfg: RetryConfig = RetryConfig()):
    score = clip_score(prompt, image, clip_model, processor, device="cpu")
    if score < cfg.clip_score_threshold:
        raise PromptImageSimilarityError(score, cfg.clip_score_threshold)
    return score


def check_text_match(prompt: str, image: Image.Image, cfg: RetryConfig = RetryConfig()):
    extracted_text = extract_text_from_image(image)
    distance = Levenshtein.distance(prompt, extracted_text)
    normalized_distance = distance / max(len(prompt), len(extracted_text), 1)
    return normalized_distance
