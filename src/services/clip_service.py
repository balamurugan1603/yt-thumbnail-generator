from PIL import Image
import torch
import torch.nn.functional as F
from functools import lru_cache
from transformers import pipeline, CLIPProcessor, CLIPModel


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

@lru_cache(maxsize=1)
def get_model_and_processor(model_id="openai/clip-vit-base-patch32", device="cpu"):
	model = CLIPModel.from_pretrained(model_id).to(device)
	processor = CLIPProcessor.from_pretrained(model_id)
	return model, processor

def clip_score(prompt: str, image: Image.Image, model, processor, device="cpu") -> float:
    inputs = processor(
        text=[prompt],
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out       = model(**inputs)
        img_emb   = out.image_embeds
        txt_emb   = out.text_embeds
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    txt_emb = F.normalize(txt_emb, p=2, dim=-1)
    scores = (txt_emb @ img_emb.T).squeeze(1)
    return scores.item()
