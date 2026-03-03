from langdetect import detect, DetectorFactory

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