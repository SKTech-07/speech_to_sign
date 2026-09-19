import os
import logging
from typing import List, Dict, Any, Optional
from app.gloss.rules import ISLRuleEngine

logger = logging.getLogger(__name__)


class GlossGenerator:
    """
    Pluggable ISL Gloss Generator.
    Supports T5 text-to-text model when configured & available,
    falling back to ISLRuleEngine.
    """

    def __init__(self, use_t5: bool = False, t5_model_name: Optional[str] = None):
        self.use_t5 = use_t5 or os.getenv("USE_T5", "false").lower() == "true"
        self.t5_model_name = t5_model_name or os.getenv("T5_MODEL_NAME", "")
        self.rule_engine = ISLRuleEngine()
        self.t5_pipeline = None

        if self.use_t5 and self.t5_model_name:
            self._load_t5_model()

    def _load_t5_model(self):
        try:
            from transformers import pipeline
            logger.info(f"Loading T5 model for English->ISL Gloss: {self.t5_model_name}")
            self.t5_pipeline = pipeline("text2text-generation", model=self.t5_model_name)
        except Exception as e:
            logger.warning(f"Could not load T5 model '{self.t5_model_name}': {e}. Falling back to Rule Engine.")
            self.t5_pipeline = None

    def generate(self, text: str, nlp_result: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Generates ISL Gloss array for input English text.
        """
        if not text or not text.strip():
            return []

        if self.t5_pipeline is not None:
            try:
                result = self.t5_pipeline(text, max_length=64)
                raw_output = result[0].get("generated_text", "")
                tokens = raw_output.split()
                return self.normalize_gloss(tokens)
            except Exception as e:
                logger.error(f"Error during T5 gloss generation: {e}. Using rule fallback.")

        # Fallback to rule engine
        raw_gloss = self.rule_engine.apply_rules(text, nlp_result)
        return self.normalize_gloss(raw_gloss)

    @staticmethod
    def normalize_gloss(raw_gloss: List[str]) -> List[str]:
        """
        Normalizes gloss tokens:
        - Uppercase
        - Strips punctuation and whitespace
        - Filters empty strings
        """
        normalized = []
        for token in raw_gloss:
            clean_token = token.strip().upper()
            # Remove any trailing punctuation attached to token
            clean_token = "".join(c for c in clean_token if c.isalnum())
            if clean_token:
                normalized.append(clean_token)
        return normalized


# Default generator instance
_default_generator = GlossGenerator()


def generate_gloss(text: str, nlp_result: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Convenience function to generate ISL Gloss array for given text and optional NLP analysis.
    """
    return _default_generator.generate(text, nlp_result)
