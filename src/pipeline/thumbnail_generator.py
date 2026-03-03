import time
from settings.config import SDXL_ID, SEED, RetryConfig
from services.llm_service import get_diffusion_input
from services.diffusion_service import generate_image
from utils.image_utils import apply_directional_gradient, render_text
from utils.lang_utils import is_english
from checks.image_checks import check_zone_clutter
from checks.prompt_checks import check_prompt_length, check_prompt_language
from exceptions.pipeline_exceptions import (
    ThumbnailPipelineError, 
    ClutterCheckError
)
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GenerationAttempt:
    seed: int
    image: object          # PIL Image
    edge_density: float
    attempt_num: int

def generate_thumbnail_pipeline(
    prompt: str,
    model_id: str = SDXL_ID,
    retry_cfg: RetryConfig = RetryConfig(),
) -> object | None:

    try:
        check_prompt_length(prompt)
        check_prompt_language(prompt)
    except ThumbnailPipelineError as e:
        logger.error("Prompt validation failed: %s", e)
        return None

    # --- Prompt enhancement ---
    spec = get_diffusion_input(prompt, seed=SEED)
    logger.info("Prompt enhanced | sdxl_prompt=%r", spec.prompt)

    attempts: list[GenerationAttempt] = []

    for attempt_num, offset in enumerate(retry_cfg.seed_offsets[: retry_cfg.max_attempts], start=1):
        attempt_seed = SEED + offset
        logger.info(
            "Image generation attempt %d/%d | seed=%d",
            attempt_num, retry_cfg.max_attempts, attempt_seed,
        )

        try:
            bg = generate_image(model_id=model_id, diffusion_input=spec, seed=attempt_seed)
            bg = apply_directional_gradient(bg, spec.text_design.text_zone)
            bg.save(f"artifacts/bg-attempt{attempt_num}-seed{attempt_seed}.png")
            clutterness = check_zone_clutter(bg, spec.text_design, cfg=retry_cfg)
        except ClutterCheckError as e:
            logger.warning("Attempt %d clutter check failed | %s", attempt_num, e)
            attempts.append(GenerationAttempt(
                seed=attempt_seed,
                image=bg,
                edge_density=e.edge_density,
                attempt_num=attempt_num,
            ))
            continue
        except ThumbnailPipelineError as e:
            logger.error("Attempt %d unrecoverable pipeline error: %s", attempt_num, e)
            return None
        
        # Only reaches here if check passed
        logger.info("Clutter check passed on attempt %d", attempt_num)
        break
    
    else:
        # All attempts exhausted without passing the threshold
        if not attempts:
            logger.error("All %d attempts failed with exceptions — aborting", retry_cfg.max_attempts)
            return None

        if retry_cfg.fallback_to_best:
            best = min(attempts, key=lambda a: a.edge_density)
            logger.warning(
                "All %d attempts exceeded clutter threshold. "
                "Falling back to best attempt (attempt=%d, edge_density=%.4f)",
                retry_cfg.max_attempts, best.attempt_num, best.edge_density,
            )
            bg = best.image
        else:
            logger.error(
                "All %d attempts exceeded clutter threshold — aborting",
                retry_cfg.max_attempts,
            )
            return None

    bg.save(f"artifacts/bg-{int(time.time())}.png")
    result = render_text(bg, prompt, spec.text_design)
    return result