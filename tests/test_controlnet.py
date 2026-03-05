import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
import os

os.environ["HF_HUB_CACHE"] = "D:\YT Thumbnail Generator\models"
os.environ["HF_TOKEN"] = "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"

CACHE_DIR = "D:\YT Thumbnail Generator\models"

original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/non-enhanced-prompt.png"
)

image = np.array(original_image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    cachec_dir=CACHE_DIR,
)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
    cachec_dir=CACHE_DIR,
)
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    cachec_dir=CACHE_DIR,
).to("cuda")

prompt = """
a relaxed rabbit sitting on a striped towel next to a pool with a tropical drink nearby, 
bright sunny day, vacation scene, 35mm photograph, film, professional, 4k, highly detailed
"""
negative_prompt = "lowres, bad anatomy, worst quality, low quality, deformed, ugly"

pipeline(
    prompt,
    negative_prompt=negative_prompt,
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
