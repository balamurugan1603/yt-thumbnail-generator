import gradio as gr
from settings.config import SDXL_ID, SEED
from services.llm_service import get_diffusion_input
from services.diffusion_service import generate_image
from utils.image_utils import apply_directional_gradient, render_text, rgb_to_sobel
from utils.lang_utils import is_english


def generate_thumbnail_pipeline(prompt: str, model_id: str = SDXL_ID):
    if len(prompt.split()) > 15:
        print("Prompt must be 15 words or fewer")
        return None
    
    if not is_english(prompt):
        print("Prompt must be in English")
        return None

    spec   = get_diffusion_input(prompt, seed=SEED)
    print(f"Generated SDXL input: {spec.prompt}\n{spec.negative_prompt}\nTextDesign: {spec.text_design}")
    bg     = generate_image(model_id=model_id, diffusion_input=spec, seed=SEED)
    bg     = apply_directional_gradient(bg, spec.text_design.text_zone)
    result = render_text(bg, prompt, spec.text_design)

    return result

# def two_step_generation_pipeline(prompt: str):
#     single_step_result, spec = generate_thumbnail_pipeline(prompt, model_id=SDXL_LIGHTNING_ID)
    
#     # TODO: Remove
#     single_step_result.save("artifacts/first_pass_output.png")

#     # single_step_result = np.array(single_step_result)
#     # sobel_result = rgb_to_sobel(single_step_result)
#     # guide_image = Image.fromarray(sobel_result)
    
#     # TODO: Remove
#     # guide_image.save("artifacts/guide_image.png")
#     spec.prompt = f"TITLE: '{prompt}' " + spec.prompt + "Foreground text and background aligns so well"
#     second_pass_result = generate_image(model_id=SD_ID, diffusion_input=spec, guide_image=single_step_result, seed=SEED)

#     # TODO: Remove
#     second_pass_result.save("artifacts/second_pass_output.png")
#     return second_pass_result



