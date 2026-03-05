from openai import OpenAI
from core.models import StableDiffusionLLMInput
from core.prompt_builder import SYSTEM_PROMPT, USER_TEMPLATE

client = OpenAI()


def get_diffusion_input(
    prompt: str, seed: int, temperature: float = 0.0
) -> StableDiffusionLLMInput:
    """Call GPT-4o to produce a structured thumbnail spec."""
    response = client.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt)},
        ],
        response_format=StableDiffusionLLMInput,
        seed=seed,
        temperature=temperature,
    )
    return response.choices[0].message.parsed
