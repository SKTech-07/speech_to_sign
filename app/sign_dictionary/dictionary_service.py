import os
import logging
from typing import Dict, Any, List, Optional
from app.cwasa.adapter import CWASAAdapter

logger = logging.getLogger(__name__)


class SignDictionary:
    """
    Generic sign dictionary service.
    Loads actual .sigml files from disk as the authoritative source of truth.
    Does NOT assume any specific sign language (ISL, ASL, etc.), nor does it fabricate HamNoSys notation.
    """

    def __init__(self, dictionary_path: Optional[str] = None):
        self.dictionary_path = dictionary_path or os.getenv("SIGN_DICTIONARY_PATH", "data/SignFiles")
        # Map normalized key -> actual file name / path
        self.index_map: Dict[str, str] = {}
        self.load_dictionary_index()

    def load_dictionary_index(self) -> None:
        """
        Scans the sign dictionary directory and builds an in-memory filename index.
        Supports data/SignFiles or data/sign_dictionary.
        """
        self.index_map.clear()
        
        # Fallback path checks if primary path does not exist
        target_path = self.dictionary_path
        if not os.path.exists(target_path):
            alt_path = "data/sign_dictionary" if target_path == "data/SignFiles" else "data/SignFiles"
            if os.path.exists(alt_path):
                target_path = alt_path

        if not os.path.exists(target_path):
            logger.warning(f"Sign dictionary directory not found at '{target_path}'.")
            return

        try:
            entries = os.listdir(target_path)
            sigml_files = [f for f in entries if f.lower().endswith(".sigml")]
            
            for fname in sigml_files:
                base_name = os.path.splitext(fname)[0]
                norm_key = base_name.strip().lower()
                full_path = os.path.join(target_path, fname)
                
                # Store normalized key -> full file path
                self.index_map[norm_key] = full_path

            logger.info(f"Indexed {len(self.index_map)} .sigml files from '{target_path}'.")
            self.dictionary_path = target_path

        except Exception as e:
            logger.error(f"Error scanning sign dictionary directory '{target_path}': {e}")

    def lookup_sign(self, query: str) -> Dict[str, Any]:
        """
        Looks up a sign in the dictionary by word/gloss term.

        Args:
            query: Input word or gloss string (e.g. "afraid", "WATER").

        Returns:
            Dictionary response object:
            - If found & valid: { "found": True, "query": ..., "matched_key": ..., "source_file": ..., "format": "sigml", "status": "READY_FOR_CWASA", "sigml": "..." }
            - If not found: { "found": False, "query": ..., "status": "SIGN_NOT_AVAILABLE" }
        """
        if not query or not query.strip():
            return {
                "found": False,
                "query": query,
                "status": "SIGN_NOT_AVAILABLE",
                "message": "Query string is empty"
            }

        norm_query = query.strip().lower()
        file_path = self.index_map.get(norm_query)

        if not file_path or not os.path.exists(file_path):
            return {
                "found": False,
                "query": query,
                "status": "SIGN_NOT_AVAILABLE",
                "message": f"Sign not available for '{query}'"
            }

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sigml_content = f.read()

            # Validate SiGML XML structure
            val_res = CWASAAdapter.validate_sigml(sigml_content)
            if not val_res["valid"]:
                logger.error(f"Invalid SiGML file '{file_path}': {val_res['error']}")
                return {
                    "found": False,
                    "query": query,
                    "source_file": file_path,
                    "status": "INVALID_SIGML",
                    "error": val_res["error"]
                }

            matched_key = os.path.splitext(os.path.basename(file_path))[0]

            return {
                "found": True,
                "query": query,
                "matched_key": matched_key,
                "source_file": file_path,
                "format": "sigml",
                "status": "READY_FOR_CWASA",
                "sigml": sigml_content
            }

        except Exception as e:
            logger.error(f"Error reading sign file '{file_path}': {e}")
            return {
                "found": False,
                "query": query,
                "source_file": file_path,
                "status": "SIGN_FILE_NOT_FOUND",
                "error": str(e)
            }

    def lookup_multiple_signs(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Looks up multiple terms in sequence.
        """
        return [self.lookup_sign(q) for q in queries]
