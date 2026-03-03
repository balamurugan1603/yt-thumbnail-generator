import time
from settings.config import SDXL_ID, SEED, RetryConfig
from services.llm_service import get_diffusion_input
from services.diffusion_service import generate_image
from utils.image_utils import apply_directional_gradient, render_text
from utils.lang_utils import is_english
from checks.image_checks import check_background_content, check_zone_clutter, check_text_contrast
from checks.prompt_checks import check_prompt_length, check_prompt_language
from exceptions.pipeline_exceptions import (
    ArtifactDetectedError,
    ThumbnailPipelineError, 
    ClutterCheckError,
    LowTextImageContrastRatioError
)
from dataclasses import dataclass, field
from PIL import Image
import logging

logger = logging.getLogger(__name__)

NUM_CHECKS = 3

@dataclass
class CheckResult:
    name: str
    passed: bool
    error: ThumbnailPipelineError | None = None
    # For tiebreaking (lower edge_density / higher contrast ratio = better)
    edge_density: float | None = None
    primary_contrast_ratio: float | None = None
    secondary_contrast_ratio: float | None = None
    artifact_probability: float | None = None

@dataclass
class GenerationAttempt:
    attempt_num: int
    seed: int
    image: Image.Image
    check_results: list[CheckResult] = field(default_factory=list)

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.check_results if c.passed)

    @property
    def failed_check(self) -> CheckResult | None:
        return next((c for c in self.check_results if not c.passed), None)

    @property
    def all_passed(self) -> bool:
        return len(self.check_results) == NUM_CHECKS and self.checks_passed == NUM_CHECKS


def _run_checks(bg, spec, retry_cfg) -> list[CheckResult]:
    results = []

    # --- Contrast ---
    try:
        ratios = check_text_contrast(bg, spec.text_design)
        results.append(CheckResult(
            name="contrast",
            passed=True,
            primary_contrast_ratio=ratios["primary"],
            secondary_contrast_ratio=ratios["secondary"],
        ))
    except LowTextImageContrastRatioError as e:
        results.append(CheckResult(
            name="contrast",
            passed=False,
            error=e,
            primary_contrast_ratio=e.primary_ratio,
            secondary_contrast_ratio=e.secondary_ratio,
        ))

    # --- Clutter ---
    try:
        clutterness = check_zone_clutter(bg, spec.text_design, cfg=retry_cfg)
        results.append(CheckResult(
            name="clutter",
            passed=True,
            edge_density=clutterness["edge_density"],
        ))
    except ClutterCheckError as e:
        results.append(CheckResult(
            name="clutter",
            passed=False,
            error=e,
            edge_density=e.edge_density,
        ))

    # --- Artifact Detection ---
    try:
        artifacts = check_background_content(bg)
        results.append(CheckResult(
            name="artifact",
            passed=True,
            artifact_probability=artifacts[0]["neg_score"],
        ))
    except ArtifactDetectedError as e:
        results.append(CheckResult(
            name="artifact",
            passed=False,
            error=e,
            artifact_probability=e.neg_score,
        ))

    return results


def _best_fallback(attempts: list[GenerationAttempt]) -> GenerationAttempt:
    """
    Pick the attempt with the most checks passed.
    Tiebreak by summing normalized scores across all checks:
      - clutter:  1 - edge_density         (lower density → closer to 1)
      - contrast: ratio / 21.0             (higher ratio  → closer to 1)
    """
    def tiebreak_score(a: GenerationAttempt) -> float:
        score = 0.0
        for c in a.check_results:
            if c.name == "clutter" and c.edge_density is not None:
                score += 1.0 - c.edge_density      # lower clutter = higher score
            elif c.name == "contrast":
                contrast_ratio_score = 0
                if c.primary_contrast_ratio is not None:
                    contrast_ratio_score += c.primary_contrast_ratio / 21.0   # normalize WCAG max ratio
                if c.secondary_contrast_ratio is not None:
                    contrast_ratio_score += c.secondary_contrast_ratio / 21.0
                score += contrast_ratio_score / 2.0  # average of primary and secondary ratios
            elif c.name == "artifact" and c.artifact_probability is not None:
                score += 1.0 - c.artifact_probability  # lower artifact probability = higher score
        return score

    return max(attempts, key=lambda a: (a.checks_passed, tiebreak_score(a)))


def generate_thumbnail_pipeline(
    prompt: str,
    model_id: str = SDXL_ID,
    retry_cfg: RetryConfig = RetryConfig(),
) -> tuple[Image.Image, GenerationAttempt] | None:

    try:
        check_prompt_length(prompt)
        check_prompt_language(prompt)
    except ThumbnailPipelineError as e:
        logger.error("Prompt validation failed: %s", e)
        return None

    spec = get_diffusion_input(prompt, seed=SEED)
    logger.info("Prompt enhanced | sdxl_prompt=%r", spec.prompt)

    attempts: list[GenerationAttempt] = []

    for attempt_num, offset in enumerate(retry_cfg.seed_offsets[:retry_cfg.max_attempts], start=1):
        attempt_seed = SEED + offset
        logger.info("Generation attempt %d/%d | seed=%d", attempt_num, retry_cfg.max_attempts, attempt_seed)

        try:
            bg = generate_image(model_id=model_id, diffusion_input=spec, seed=attempt_seed)
            bg = apply_directional_gradient(bg, spec.text_design.text_zone)
            bg.save(f"artifacts/bg-attempt{attempt_num}-seed{attempt_seed}.png")
        except ThumbnailPipelineError as e:
            logger.error("Attempt %d image generation failed: %s", attempt_num, e)
            return None

        check_results = _run_checks(bg, spec, retry_cfg)
        attempt = GenerationAttempt(
            attempt_num=attempt_num,
            seed=attempt_seed,
            image=bg,
            check_results=check_results,
        )
        attempts.append(attempt)

        logger.info(
            "Attempt %d | checks_passed=%d/%d | failed_on=%s",
            attempt_num,
            attempt.checks_passed,
            NUM_CHECKS,
            attempt.failed_check.name if attempt.failed_check else "none",
        )

        if attempt.all_passed:
            logger.info("All checks passed on attempt %d", attempt_num)
            break
    else:
        if retry_cfg.fallback_to_best:
            attempt = _best_fallback(attempts)
            logger.warning(
                "All %d attempts failed. Falling back to attempt %d "
                "(checks_passed=%d/%d, failed_on=%s)",
                retry_cfg.max_attempts,
                attempt.attempt_num,
                attempt.checks_passed,
                NUM_CHECKS,
                attempt.failed_check.name if attempt.failed_check else "none",
            )
            bg = attempt.image
        else:
            logger.error("All %d attempts failed — aborting", retry_cfg.max_attempts)
            return None

    return (render_text(bg, prompt, spec.text_design), attempt)