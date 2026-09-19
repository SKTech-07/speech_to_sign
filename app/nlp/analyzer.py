import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_nlp_model = None

NEGATION_LEMMAS = {"not", "no", "never", "cannot", "neither", "nor", "nothing", "nobody", "none"}


def get_nlp_model(model_name: Optional[str] = "en_core_web_sm"):
    """
    Lazy loads the spaCy NLP model.
    """
    global _nlp_model
    if _nlp_model is None:
        import spacy
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            _nlp_model = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            _nlp_model = spacy.load(model_name)
    return _nlp_model


def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyzes English text using spaCy and extracts tokens, lemmas, POS tags,
    dependency relations, named entities, and negation signals.

    Args:
        text: Preprocessed English text string.

    Returns:
        Structured dictionary matching NLPAnalysisResult schema:
        {
          "tokens": [...],
          "lemmas": [...],
          "pos": [...],
          "dependencies": [{"token": ..., "dep": ..., "head": ...}],
          "entities": [{"text": ..., "label": ...}],
          "negation": [...]
        }
    """
    if not text or not text.strip():
        return {
            "tokens": [],
            "lemmas": [],
            "pos": [],
            "dependencies": [],
            "entities": [],
            "negation": []
        }

    nlp = get_nlp_model()
    doc = nlp(text)

    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    pos_tags = [token.pos_ for token in doc]
    
    dependencies = [
        {"token": token.text, "dep": token.dep_, "head": token.head.text}
        for token in doc
    ]

    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]

    negations = [
        token.text for token in doc
        if token.lemma_.lower() in NEGATION_LEMMAS or token.dep_ == "neg"
    ]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos": pos_tags,
        "dependencies": dependencies,
        "entities": entities,
        "negation": negations
    }
