from openai import OpenAI
from core.models import StableDiffusionInput
from core.prompt_builder import SYSTEM_PROMPT, USER_TEMPLATE

client = OpenAI()

def get_diffusion_input(prompt: str, seed: int) -> StableDiffusionInput:
    """Call GPT-4o to produce a structured thumbnail spec."""
    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(prompt=prompt)},
        ],
        response_format=StableDiffusionInput,
        seed=seed,
    )
    return response.choices[0].message.parsed