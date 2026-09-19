import os
import re
import logging
from typing import List, Tuple
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Fallback common glosses if SignFiles dir is empty
DEFAULT_COMMON_GLOSSES = [
    "afraid", "water", "school", "name", "hello", "help",
    "thank you", "drink", "eat", "please", "yes", "no", "want", "like"
]

_cached_gloss_list: List[str] = []
_cached_initial_prompt: str = ""

def load_dictionary_glosses(sign_files_dir: str = "data/SignFiles") -> List[str]:
    """
    Scans the sign directory to collect known gloss words.
    """
    global _cached_gloss_list
    if _cached_gloss_list:
        return _cached_gloss_list

    glosses = set(DEFAULT_COMMON_GLOSSES)
    
    if os.path.exists(sign_files_dir):
        for fname in os.listdir(sign_files_dir):
            if fname.endswith(".sigml"):
                gloss = fname[:-6].lower()
                glosses.add(gloss)
                
    _cached_gloss_list = sorted(list(glosses))
    logger.info(f"Loaded {len(_cached_gloss_list)} dictionary glosses for STT post-processing.")
    return _cached_gloss_list


def build_initial_prompt(sign_files_dir: str = "data/SignFiles", max_len: int = 200) -> str:
    """
    Builds a domain vocabulary initial_prompt string capped at ~200 characters.
    """
    global _cached_initial_prompt
    if _cached_initial_prompt:
        return _cached_initial_prompt

    glosses = load_dictionary_glosses(sign_files_dir)
    
    prompt_words = []
    current_len = 0
    
    for word in glosses:
        w_str = word if not prompt_words else f", {word}"
        if current_len + len(w_str) > max_len:
            break
        prompt_words.append(word)
        current_len += len(w_str)
        
    _cached_initial_prompt = ", ".join(prompt_words)
    logger.info(f"Built STT initial_prompt ({len(_cached_initial_prompt)} chars): '{_cached_initial_prompt}'")
    return _cached_initial_prompt


def correct_transcript(raw_text: str, threshold: float = 88.0) -> Tuple[str, bool]:
    """
    Lowercases and strips the transcript, then fuzzy matches each token against
    known dictionary glosses using rapidfuzz token_sort_ratio.
    
    If a token scores >= 88 against exactly one gloss, substitutes the gloss.
    Returns (corrected_text, is_speech_detected).
    """
    if not raw_text:
        return "", False
        
    cleaned_text = raw_text.strip().lower()
    
    # Strip non-alphanumeric punctuation except space
    word_tokens = re.findall(r"\b\w+\b", cleaned_text)
    
    if not word_tokens:
        logger.info(f"Transcript contains no valid words (raw: '{raw_text}') -> NO_SPEECH_DETECTED")
        return "", False
        
    gloss_list = load_dictionary_glosses()
    corrected_tokens = []
    
    for token in word_tokens:
        # Check exact match first
        if token in gloss_list:
            corrected_tokens.append(token)
            continue
            
        matches = []
        for gloss in gloss_list:
            score = fuzz.token_sort_ratio(token, gloss)
            if score >= threshold:
                matches.append((score, gloss))
                
        if len(matches) == 1:
            best_gloss = matches[0][1]
            logger.info(f"Fuzzy match corrected token '{token}' -> '{best_gloss}' (score: {matches[0][0]})")
            corrected_tokens.append(best_gloss)
        elif len(matches) > 1:
            # Sort by score descending
            matches.sort(key=lambda x: x[0], reverse=True)
            if matches[0][0] > matches[1][0]:
                best_gloss = matches[0][1]
                logger.info(f"Fuzzy match resolved tie token '{token}' -> '{best_gloss}' (score: {matches[0][0]})")
                corrected_tokens.append(best_gloss)
            else:
                corrected_tokens.append(token)
        else:
            corrected_tokens.append(token)
            
    corrected_text = " ".join(corrected_tokens)
    return corrected_text, True
