from exceptions.pipeline_exceptions import PromptTooLongError, NonEnglishPromptError
from utils.lang_utils import is_english
from settings.config import WORD_LIMIT

def check_prompt_length(prompt: str, limit: int = WORD_LIMIT):
    word_count = len(prompt.split())
    if word_count > limit:
        raise PromptTooLongError(word_count, limit)

def check_prompt_language(prompt: str):
    if not is_english(prompt):
        raise NonEnglishPromptError(prompt)
