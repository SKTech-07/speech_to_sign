import re

# Comprehensive contraction mapping preserving meaning, especially negation
CONTRACTIONS = {
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot",
    r"\bcannot\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bi'm\b": "I am",
    r"\byou're\b": "you are",
    r"\bhe's\b": "he is",
    r"\bshe's\b": "she is",
    r"\bit's\b": "it is",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\bi'll\b": "I will",
    r"\byou'll\b": "you will",
    r"\bi've\b": "I have",
    r"\blet's\b": "let us",
}

# Essential words that must NEVER be deleted during preprocessing
PRESERVED_NEGATION_WORDS = {"not", "no", "never", "cannot", "none", "nobody", "nothing", "neither", "nor"}


def preprocess_text(text: str) -> str:
    """
    Preprocesses raw English text for NLP analysis and ISL Gloss conversion.

    Steps:
    1. Strip leading/trailing whitespace & normalize internal whitespace.
    2. Expand contractions (e.g. "don't" -> "do not"), explicitly preserving negation.
    3. Remove non-semantic punctuation (!, ., ,, etc.) while keeping question marks or essential tokens.
    4. Normalize capitalization while maintaining readability.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized English string.
    """
    if not text:
        return ""

    # 1. Normalize whitespace
    cleaned = " ".join(text.strip().split())

    # 2. Expand contractions (case-insensitive regex substitution)
    for pattern, replacement in CONTRACTIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Preserve question mark as a distinct token or marker if present
    is_question = "?" in cleaned

    # 3. Remove extraneous punctuation (keep letters, digits, spaces, and ?)
    cleaned = re.sub(r"[^\w\s\?]", "", cleaned)

    # 4. Clean spaces again after punctuation removal
    cleaned = " ".join(cleaned.split())

    return cleaned
