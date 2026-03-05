from PIL import Image
import torch
import torch.nn.functional as F
from functools import lru_cache
from transformers import CLIPProcessor, CLIPModel

@lru_cache(maxsize=1)
def get_model_and_processor(model_id="openai/clip-vit-base-patch32", device="cpu"):
	model = CLIPModel.from_pretrained(model_id).to(device)
	processor = CLIPProcessor.from_pretrained(model_id)
	return model, processor

def clip_score(prompt: str, image: Image.Image, device="cpu") -> float:
    model, processor = get_model_and_processor(device=device)
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

def test_clip_score(prompts, image_paths):
    scores = []
    for prompt, path in zip(prompts, image_paths):
        img = Image.open(path)
        score = clip_score(prompt, img)
        scores.append(score)
        print(f"CLIP score: {score}")


prompts_images = [
    ("Why Most Startups Fail", "src/artifacts/bg-attempt1-seed22.png"),
    ("I Tested AI for 30 Days", "src/artifacts/bg-attempt4-seed53.png"),
    ("Machine Learning Workshop", "src/artifacts/bg-attempt3-seed35.png"),
    ("Only 1% can solve this riddle", "src/artifacts/bg-attempt2-seed29.png"),
]

if __name__ == "__main__":
    prompts, image_paths = zip(*prompts_images)
    test_clip_score(prompts, image_paths)
