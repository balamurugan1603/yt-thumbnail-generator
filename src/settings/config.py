import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

SEED = 22
TEMPERATURE = 0.0
MAX_CONCURRENCY = 5

CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INFERENCE_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/aa4b220c43c7a332da20e3d3de64ad62/ai/run/"
SDXL_LIGHTNING_ID = "@cf/bytedance/stable-diffusion-xl-lightning"
SDXL_ID = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
SD_ID = "@cf/runwayml/stable-diffusion-v1-5-img2img"
CLOUDFLARE_HEADERS = {"Authorization": f"Bearer {CLOUDFLARE_API_KEY}"}

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
WORD_LIMIT = 15

WCAG_AA_LARGE = 3.0


@dataclass
class RetryConfig:
    max_attempts: int = 4
    clutter_threshold: float = 0.075
    contrast_ratio_threshold: float = WCAG_AA_LARGE
    clip_score_threshold: float = 0.2
    # Each attempt uses a different seed offset to get varied generations
    # seed_offsets: list[int] = field(default_factory=lambda: [0, 7, 13, 31])
    # If all retries fail, fall back to the least-cluttered attempt
    fallback_to_best: bool = True
