import re
from typing import List, Dict, Any, Optional

# Words that do not carry independent sign meaning in ISL
STOP_WORDS = {
    "a", "an", "the", "to", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "of", "for", "in", "on", "at", "by", "with"
}

# Modal and desire verbs that typically follow the main action verb in ISL
DESIRE_MODAL_VERBS = {"want", "wants", "wanted", "can", "could", "must", "should", "would", "need", "needs", "like", "likes"}

# Question words
QUESTION_WORDS = {"what", "where", "who", "when", "why", "how", "which", "whose", "whom"}

# Negation words
NEGATION_WORDS = {"not", "no", "never", "cannot", "none", "nobody", "nothing"}

# Time markers
TIME_WORDS = {"today", "yesterday", "tomorrow", "now", "morning", "afternoon", "evening", "night", "soon", "later", "everyday", "daily"}

# Politeness tokens
POLITE_WORDS = {"please", "kindly", "thanks", "thank"}


class ISLRuleEngine:
    """
    Rule-based converter for English sentences -> ISL Gloss sequences.
    Enforces Subject-Object-Verb (SOV) structure, WH-words at end,
    modal verbs following action verbs, negation at sentence end, and question markers.
    """

    def apply_rules(self, text: str, nlp_result: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Converts text to an ordered ISL Gloss token list using linguistic rules.

        Args:
            text: Preprocessed English sentence.
            nlp_result: Structured spaCy analysis dict containing tokens, lemmas, pos, dep.

        Returns:
            List of uppercase ISL Gloss strings.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        has_question_mark = "?" in clean_text
        # Clean text for tokenization
        words = re.findall(r"\b\w+\b", clean_text.lower())

        if not words:
            return []

        # Categorize tokens into ISL grammatical buckets
        polite: List[str] = []
        time_tokens: List[str] = []
        subjects: List[str] = []
        objects_locations: List[str] = []
        main_verbs: List[str] = []
        desire_modals: List[str] = []
        negation: List[str] = []
        question_tokens: List[str] = []
        other_semantic: List[str] = []

        is_wh_question = any(w in QUESTION_WORDS for w in words)
        is_yes_no_question = has_question_mark and not is_wh_question

        for word in words:
            if word in POLITE_WORDS:
                polite.append(word.upper())
            elif word in QUESTION_WORDS:
                question_tokens.append(word.upper())
            elif word in NEGATION_WORDS:
                negation.append("NOT" if word in ("not", "cannot", "no") else word.upper())
            elif word in TIME_WORDS:
                time_tokens.append(word.upper())
            elif word in DESIRE_MODAL_VERBS:
                desire_modals.append("WANT" if word in ("want", "wants", "wanted") else word.upper())
            elif word in STOP_WORDS:
                continue
            else:
                # Use spaCy POS info if available to categorize pronouns, nouns, verbs
                pos = self._get_pos_for_word(word, nlp_result)
                lemma = self._get_lemma_for_word(word, nlp_result).upper()

                if word in ("i", "me", "my"):
                    # Standardize self pronoun to I or ME based on context
                    if word == "me" and polite:
                        subjects.append("ME")
                    else:
                        subjects.append("I")
                elif word in ("you", "your", "yours"):
                    if word == "your":
                        subjects.append("YOUR")
                    else:
                        subjects.append("YOU")
                elif pos in ("VERB", "AUX"):
                    main_verbs.append(lemma)
                elif pos in ("NOUN", "PROPN", "PRON", "ADJ"):
                    objects_locations.append(lemma)
                else:
                    other_semantic.append(lemma)

        # Build ISL Gloss Order:
        # [POLITE] + [SUBJECTS] + [OBJECTS/LOCATIONS] + [MAIN_VERBS] + [TIME] + [DESIRE/MODALS] + [OTHER] + [NEGATION] + [QUESTION]
        
        gloss_tokens: List[str] = []
        
        # 1. Politeness (e.g. PLEASE)
        gloss_tokens.extend(polite)

        # 2. Subject(s)
        gloss_tokens.extend(subjects)

        # 3. Objects / Locations / Complements (e.g. WATER, SCHOOL, HOSPITAL, NAME)
        gloss_tokens.extend(objects_locations)

        # 4. Main Action Verbs (e.g. DRINK, GO, EAT, GIVE)
        gloss_tokens.extend(main_verbs)

        # 5. Time expressions (e.g. TODAY) if present after action
        gloss_tokens.extend(time_tokens)

        # 6. Desire & Modal Verbs (e.g. WANT, CAN)
        gloss_tokens.extend(desire_modals)

        # 7. Other remaining semantic content
        gloss_tokens.extend(other_semantic)

        # 8. Negation at sentence/clause end (e.g. NOT)
        gloss_tokens.extend(negation)

        # 9. Question tokens / WH-words at end
        gloss_tokens.extend(question_tokens)
        if is_yes_no_question and "QUESTION" not in gloss_tokens:
            gloss_tokens.append("QUESTION")

        # Deduplicate consecutive duplicates while preserving order
        deduped: List[str] = []
        for t in gloss_tokens:
            if not deduped or deduped[-1] != t:
                deduped.append(t)

        return deduped

    def _get_pos_for_word(self, word: str, nlp_result: Optional[Dict[str, Any]]) -> str:
        if not nlp_result:
            return ""
        tokens = nlp_result.get("tokens", [])
        pos_list = nlp_result.get("pos", [])
        for t, p in zip(tokens, pos_list):
            if t.lower() == word.lower():
                return p
        return ""

    def _get_lemma_for_word(self, word: str, nlp_result: Optional[Dict[str, Any]]) -> str:
        if not nlp_result:
            return word
        tokens = nlp_result.get("tokens", [])
        lemmas = nlp_result.get("lemmas", [])
        for t, l in zip(tokens, lemmas):
            if t.lower() == word.lower():
                return l if l != "-PRON-" else word
        return word
