import requests
from io import BytesIO
from PIL import Image
from settings.config import INFERENCE_ENDPOINT, CLOUDFLARE_HEADERS, IMAGE_WIDTH, IMAGE_HEIGHT
from core.models import StableDiffusionInput, StableDiffusionLLMInput, TextDesign
from utils.image_utils import pil_to_base64

def generate_image(model_id: str, diffusion_input: StableDiffusionLLMInput | StableDiffusionInput, guide_image: Image.Image | None = None, seed: int = 0) -> Image.Image:
    """Call Cloudflare SDXL to generate the background image."""
    payload = {
        "prompt":          diffusion_input.prompt,
        "negative_prompt": diffusion_input.negative_prompt,
        "width":           IMAGE_WIDTH,
        "height":          IMAGE_HEIGHT,
        "guidance":        7.5,
        "seed":            seed,
    }
    
    if guide_image is not None:
        # Resize to target dimensions before encoding
        resized = guide_image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        payload["image_b64"] = pil_to_base64(resized)
        payload["strength"] = 0.3

    response = requests.post(
        INFERENCE_ENDPOINT + model_id,
        headers=CLOUDFLARE_HEADERS,
        json=payload,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Cloudflare API Error: {response.text}")

    return Image.open(BytesIO(response.content)).convert("RGB")

def create_stable_diffusion_input(prompt: str, negative_prompt: str) -> StableDiffusionInput:
    """Construct StableDiffusionInput from prompt and text design."""
    return StableDiffusionInput(
        prompt=prompt,
        negative_prompt=negative_prompt
    )