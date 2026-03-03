from pydantic import BaseModel, Field
from typing import List

class TextDesign(BaseModel):
    font_style:     str          = Field(..., description="Font personality: bold, cinematic, playful, etc.")
    primary_color:  str          = Field(..., description="Main text color in HEX.")
    secondary_color: str         = Field(..., description="Emphasis word color in HEX.")
    primary_stroke_color: str    = Field(..., description="Stroke/outline color in HEX.")
    secondary_stroke_color: str  = Field(..., description="Emphasis word stroke color in HEX.")
    stroke_width:   int          = Field(..., description="Stroke thickness in pixels.")
    shadow_color:   str          = Field(..., description="Drop shadow color in HEX.")
    shadow_offset:  List[int]    = Field(..., description="Shadow [x, y] offset in pixels.")
    text_zone:      str          = Field(..., description="Where to place text: 'left', 'top-band', 'bottom-band', 'center'.")
    emphasis_words: List[str]    = Field(..., description="Words to visually emphasize.")
    emphasis_scale: float        = Field(..., description="Scale multiplier for emphasis words (e.g. 1.2).")

class StableDiffusionLLMInput(BaseModel):
    prompt:          str        = Field(..., description="Optimized SDXL background prompt.")
    negative_prompt: str        = Field(..., description="Negative prompt to avoid text/artifacts.")
    text_design:     TextDesign = Field(..., description="Text styling spec.")

class StableDiffusionInput(BaseModel):
    prompt:          str        = Field(..., description="Optimized SDXL background prompt.")
    negative_prompt: str        = Field(..., description="Negative prompt to avoid text/artifacts.")
