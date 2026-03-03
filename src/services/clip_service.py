import torch
from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def load_clip_zero_shot():
    device = (
        0      if torch.cuda.is_available()         else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    return pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        dtype=torch.bfloat16,
        device=device,
    )