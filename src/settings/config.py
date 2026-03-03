import os
from dotenv import load_dotenv

load_dotenv()

SEED = 22
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")

INFERENCE_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/aa4b220c43c7a332da20e3d3de64ad62/ai/run/"
SDXL_LIGHTNING_ID = "@cf/bytedance/stable-diffusion-xl-lightning"
SDXL_ID  = "@cf/stabilityai/stable-diffusion-xl-base-1.0"
SD_ID              = "@cf/runwayml/stable-diffusion-v1-5-img2img"
CLOUDFLARE_HEADERS = {"Authorization": f"Bearer {CLOUDFLARE_API_KEY}"}

IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 720