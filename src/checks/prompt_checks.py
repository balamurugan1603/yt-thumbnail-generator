from exceptions.pipeline_exceptions import PromptTooLongError, NonEnglishPromptError
from utils.lang_utils import is_english_lingua
from settings.config import WORD_LIMIT
import re

def check_prompt_length(prompt: str, limit: int = WORD_LIMIT):
    word_count = len(prompt.split())
    if word_count > limit:
        raise PromptTooLongError(word_count, limit)

def check_prompt_language(prompt: str):
    prompt = re.sub(r'[^a-zA-Z0-9]', '', prompt)
    # Only check for non-English if the prompt is longer than 3 words to avoid false positives on very short prompts
    if not is_english_lingua(prompt) and len(prompt.split()) > 3:
        raise NonEnglishPromptError(prompt)
