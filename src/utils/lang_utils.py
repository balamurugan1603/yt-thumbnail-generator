from langdetect import detect, DetectorFactory
from lingua import Language, LanguageDetectorBuilder

def is_english_lingua(sentence):
    """
    Checks if the detected language of a sentence is English using Lingua.

    Args:
        sentence (str): The text to check.
    Returns:
        bool: True if English, False otherwise.
    """
    try:
        languages = [Language.ENGLISH]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()
        detected_language = detector.detect_language_of(sentence)
        return detected_language == Language.ENGLISH
    except Exception:
        # Handle cases where language detection might fail (e.g., empty string, gibberish)
        return False

def is_english(sentence):
    """
    Checks if the detected language of a sentence is English.

    Args:
        sentence (str): The text to check.

    Returns:
        bool: True if English, False otherwise.
    """
    try:
        # Detect the language and check if the code is 'en' (English)
        return detect(sentence) == 'en'
    except Exception:
        # Handle cases where language detection might fail (e.g., empty string, gibberish)
        return False