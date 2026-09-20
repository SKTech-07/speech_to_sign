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
    Builds a clean domain vocabulary initial_prompt string using real English words.
    """
    global _cached_initial_prompt
    if _cached_initial_prompt:
        return _cached_initial_prompt

    glosses = load_dictionary_glosses(sign_files_dir)
    # Filter for alphabetic words only to avoid numeric prompt bias (0, 1, 10, 100...)
    alpha_glosses = [g for g in glosses if g.isalpha()]
    
    prompt_words = []
    current_len = 0
    
    for word in alpha_glosses:
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
    Cleans up and optionally corrects minor typos in transcript against dictionary glosses.
    Preserves exact recognized speech content without dropping spoken words.
    Returns (corrected_text, is_speech_detected).
    """
    if not raw_text or not raw_text.strip():
        return "", False
        
    raw_clean = raw_text.strip()
    # Ensure raw_clean contains actual printable content
    if not re.search(r"[a-zA-Z0-9]", raw_clean):
        return "", False

    gloss_list = load_dictionary_glosses()
    words = raw_clean.split()
    corrected_words = []
    
    for word in words:
        # Match word prefix/punctuation, core token, and suffix/punctuation
        match = re.match(r"^([^\w]*)([\w'-]+)([^\w]*)$", word)
        if not match:
            corrected_words.append(word)
            continue
            
        prefix, token, suffix = match.groups()
        token_lower = token.lower()
        
        # Keep exact token if in gloss list or short word (<= 3 chars)
        if token_lower in gloss_list or len(token_lower) <= 3:
            corrected_words.append(word)
            continue
            
        # Optional fuzzy match check for minor typos
        matches = []
        for gloss in gloss_list:
            if abs(len(token_lower) - len(gloss)) <= 2:
                score = fuzz.token_sort_ratio(token_lower, gloss)
                if score >= threshold:
                    matches.append((score, gloss))
                    
        if len(matches) == 1:
            best_gloss = matches[0][1]
            if token.istitle():
                best_gloss = best_gloss.capitalize()
            elif token.isupper():
                best_gloss = best_gloss.upper()
            corrected_words.append(f"{prefix}{best_gloss}{suffix}")
        elif len(matches) > 1:
            matches.sort(key=lambda x: x[0], reverse=True)
            if matches[0][0] > matches[1][0]:
                best_gloss = matches[0][1]
                if token.istitle():
                    best_gloss = best_gloss.capitalize()
                elif token.isupper():
                    best_gloss = best_gloss.upper()
                corrected_words.append(f"{prefix}{best_gloss}{suffix}")
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
            
    final_text = " ".join(corrected_words)
    return final_text, True

