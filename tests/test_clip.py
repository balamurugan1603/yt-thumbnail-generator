"""
CLIP prompt pair tester — compares positive vs negative prompt probabilities.
Run: python test_clip.py
"""

import sys
import time
import requests
from io import BytesIO
from PIL import Image

print("Checking dependencies...")
try:
    import torch
    from transformers import pipeline
    print(f"  torch        {torch.__version__}")
    print(f"  transformers OK")
except ImportError as e:
    print(f"\n  Missing: {e}")
    print("  Run: pip install torch transformers pillow requests")
    sys.exit(1)

# -- Device -------------------------------------------------------------------
device = (
    0      if torch.cuda.is_available()         else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}\n")

# -- Load ---------------------------------------------------------------------
t0 = time.time()
clip = pipeline(
    task="zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    dtype=torch.bfloat16,
    device=device,
)
print(f"Loaded in {time.time() - t0:.1f}s\n")

# -- Prompt pairs -------------------------------------------------------------
# Each pair: (positive, negative) — CLIP scores both, winner determines verdict
PROMPT_PAIRS = [
    ("an image without face, hand, human artifacts, text, typography or any symbol",     "an image with face, hand, human artifacts, text, typography or any symbol")
]

# -- Fetch helper -------------------------------------------------------------
def fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

# -- Test images --------------------------------------------------------------
TEST_IMAGES = [
    ("Clean photo (COCO cats)",    "https://images.cocodataset.org/val2017/000000039769.jpg"),
    ("Clean photo (COCO baseball)","https://images.cocodataset.org/val2017/000000397133.jpg"),
    ("Lady in garden","https://cdn.pixabay.com/photo/2026/02/14/17/53/latif_photo88-lavender-10123458_1280.jpg"),  # same image, but we will test if CLIP can detect artifacts in it
]

# -- Score --------------------------------------------------------------------
all_labels = [label for pair in PROMPT_PAIRS for label in pair]  # flat list

for img_label, url in TEST_IMAGES:
    print("=" * 65)
    print(f"Image: {img_label}")
    print("=" * 65)

    try:
        image = fetch_image(url)
        print(f"Size  : {image.size[0]}x{image.size[1]}px\n")
    except Exception as e:
        print(f"Could not fetch: {e}\n")
        continue

    t0      = time.time()
    results = clip(image, candidate_labels=all_labels)
    elapsed = time.time() - t0
    scores  = {r["label"]: r["score"] for r in results}

    print(f"{'POSITIVE':<40}  {'pos':>6}  {'neg':>6}  {'winner':<10}")
    print(f"{'NEGATIVE':<40}")
    print("-" * 65)

    pair_verdicts = []
    for pos, neg in PROMPT_PAIRS:
        pos_score = scores[pos]
        neg_score = scores[neg]
        winner    = "CLEAN  OK" if pos_score > neg_score else "ARTIFACT X"
        pair_verdicts.append(pos_score > neg_score)

        print(f"+ {pos:<38}  {pos_score:.4f}")
        print(f"- {neg:<38}  {neg_score:.4f}  {winner}")
        print()

    clean_pairs = sum(pair_verdicts)
    total_pairs = len(pair_verdicts)
    print(f"Pairs won by positive : {clean_pairs}/{total_pairs}")
    print(f"Inference time        : {elapsed:.2f}s")
    print()

print("Done.")