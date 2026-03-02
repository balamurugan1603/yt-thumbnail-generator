import gradio as gr
from settings.config import SEED
from services.llm_service import get_diffusion_input
from services.diffusion_service import generate_background
from utils.image_utils import apply_directional_gradient, render_text


def generate_thumbnail_pipeline(prompt: str):
    if len(prompt.split()) > 15:
        return None, "Prompt must be 15 words or fewer"

    spec   = get_diffusion_input(prompt, SEED)
    bg     = generate_background(spec, SEED)
    bg     = apply_directional_gradient(bg, spec.text_design.text_zone)
    result = render_text(bg, prompt, spec.text_design)

    return result