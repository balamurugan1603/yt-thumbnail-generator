import requests
from io import BytesIO
from PIL import Image
from settings.config import INFERENCE_ENDPOINT, SDXL_LIGHTNING_ID, SDXL_HEADERS, IMAGE_WIDTH, IMAGE_HEIGHT
from core.models import StableDiffusionInput

def generate_background(diffusion_input: StableDiffusionInput, seed: int) -> Image.Image:
    """Call Cloudflare SDXL to generate the background image."""
    payload = {
        "prompt":          diffusion_input.prompt,
        "negative_prompt": diffusion_input.negative_prompt,
        "width":           IMAGE_WIDTH,
        "height":          IMAGE_HEIGHT,
        "guidance":        7.5,
        "seed":            seed,
    }
    response = requests.post(
        INFERENCE_ENDPOINT + SDXL_LIGHTNING_ID,
        headers=SDXL_HEADERS,
        json=payload,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Cloudflare API Error: {response.text}")

    return Image.open(BytesIO(response.content)).convert("RGB")