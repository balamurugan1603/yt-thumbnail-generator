"""
Batch inference for generate_thumbnail_pipeline.
Runs up to 10 pipelines concurrently using a semaphore-bounded thread pool.

Usage:
    results = asyncio.run(generate_thumbnails_batch(prompts))
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from PIL import Image

from settings.config import MAX_CONCURRENCY, SDXL_ID, RetryConfig
from pipeline.solo_pipeline import GenerationAttempt, generate_thumbnail_pipeline

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    prompt: str
    image: Image.Image | None
    success: bool
    error: str | None = None
    duration: float = 0.0
    best_attempt: GenerationAttempt | None = None


@dataclass
class BatchSummary:
    total: int
    succeeded: int
    failed: int
    duration: float
    results: list[BatchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.succeeded / self.total if self.total else 0.0


async def _run_single(
    prompt: str,
    model_id: str,
    retry_cfg: RetryConfig,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    index: int,
    total: int,
) -> BatchResult:
    """
    Acquires semaphore slot, runs the blocking pipeline in a thread,
    releases slot when done regardless of outcome.
    """
    async with semaphore:
        logger.info("Starting [%d/%d] | prompt=%r", index, total, prompt)
        t0 = time.perf_counter()

        loop = asyncio.get_running_loop()
        try:
            image, attempt = await loop.run_in_executor(
                executor,
                lambda: generate_thumbnail_pipeline(
                    prompt=prompt,
                    model_id=model_id,
                    retry_cfg=retry_cfg,
                ),
            )

            duration = time.perf_counter() - t0

            if image is None:
                logger.warning(
                    "Failed  [%d/%d] | pipeline returned None | prompt=%r",
                    index,
                    total,
                    prompt,
                )
                return BatchResult(
                    prompt=prompt,
                    image=None,
                    success=False,
                    error="Pipeline returned None",
                    duration=duration,
                    best_attempt=attempt,
                )

            logger.info(
                "Done    [%d/%d] | duration=%.2fs | prompt=%r",
                index,
                total,
                duration,
                prompt,
            )
            return BatchResult(
                prompt=prompt,
                image=image,
                success=True,
                duration=duration,
                best_attempt=attempt,
            )

        except Exception as e:
            duration = time.perf_counter() - t0
            logger.error("Error   [%d/%d] | %s | prompt=%r", index, total, e, prompt)
            return BatchResult(
                prompt=prompt,
                image=None,
                success=False,
                error=str(e),
                duration=duration,
            )


async def generate_thumbnails_batch(
    prompts: list[str],
    model_id: str = SDXL_ID,
    retry_cfg: RetryConfig = RetryConfig(),
    max_concurrency: int = MAX_CONCURRENCY,
) -> BatchSummary:
    """
    Generate thumbnails for all prompts concurrently, bounded to max_concurrency.

    Args:
        prompts:         List of prompts to process.
        model_id:        Diffusion model ID, defaults to SDXL_ID.
        retry_cfg:       Retry configuration shared across all tasks.
        max_concurrency: Max pipelines running in parallel (default 10).

    Returns:
        BatchSummary with per-prompt results and aggregate stats.
    """
    if not prompts:
        logger.warning("Empty prompt list passed to batch pipeline")
        return BatchSummary(total=0, succeeded=0, failed=0, duration=0.0)

    total = len(prompts)
    semaphore = asyncio.Semaphore(max_concurrency)

    # ThreadPoolExecutor sized to concurrency — pipeline is CPU/IO bound, not async-native
    executor = ThreadPoolExecutor(
        max_workers=max_concurrency, thread_name_prefix="thumbnail"
    )

    logger.info("Batch started | total=%d | max_concurrency=%d", total, max_concurrency)
    t0 = time.perf_counter()
    coroutines = [
        _run_single(
            prompt=prompt,
            model_id=model_id,
            retry_cfg=retry_cfg,
            semaphore=semaphore,
            executor=executor,
            index=i + 1,
            total=total,
        )
        for i, prompt in enumerate(prompts)
    ]

    try:
        results = await asyncio.gather(*coroutines, return_exceptions=False)
    finally:
        executor.shutdown(wait=True)

    succeeded = sum(1 for r in results if r.success)
    duration = time.perf_counter() - t0

    logger.info(
        "Batch complete | total=%d | succeeded=%d | failed=%d | duration=%.2fs | success_rate=%.1f%%",
        total,
        succeeded,
        total - succeeded,
        duration,
        (succeeded / total) * 100,
    )

    return BatchSummary(
        total=total,
        succeeded=succeeded,
        failed=total - succeeded,
        duration=duration,
        results=results,
    )
