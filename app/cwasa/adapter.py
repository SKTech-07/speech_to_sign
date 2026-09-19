import xml.etree.ElementTree as ET
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CWASAAdapter:
    """
    Modular boundary adapter for preparing valid SiGML content for CWASA player execution.
    Does NOT require a 3D avatar to be attached.
    """

    @staticmethod
    def validate_sigml(sigml_content: str) -> Dict[str, Any]:
        """
        Validates whether a given string is well-formed SiGML XML.

        Returns:
            Dict containing:
                - valid: bool
                - error: Optional[str]
        """
        if not sigml_content or not sigml_content.strip():
            return {"valid": False, "error": "EMPTY_SIGML"}

        try:
            # Parse XML tree
            root = ET.fromstring(sigml_content.strip())
            
            # Check if root tag is sigml or hns_sign
            tag_name = root.tag.lower()
            if tag_name not in ("sigml", "hns_sign"):
                return {"valid": False, "error": f"INVALID_ROOT_TAG_{root.tag}"}

            # Check if at least one hns_sign element exists
            has_hns_sign = (tag_name == "hns_sign") or (root.find(".//hns_sign") is not None)
            if not has_hns_sign:
                return {"valid": False, "error": "MISSING_HNS_SIGN_ELEMENT"}

            return {"valid": True, "error": None}

        except ET.ParseError as pe:
            logger.error(f"SiGML XML Parse Error: {pe}")
            return {"valid": False, "error": f"XML_PARSE_ERROR: {str(pe)}"}
        except Exception as e:
            logger.error(f"Unexpected error validating SiGML: {e}")
            return {"valid": False, "error": f"VALIDATION_ERROR: {str(e)}"}

    @staticmethod
    def prepare_cwasa_sequence(sigml_strings: List[str]) -> str:
        """
        Combines multiple valid SiGML sign definitions into a single, contiguous
        <sigml> XML document sequence ready for CWASA playback.

        Args:
            sigml_strings: List of individual SiGML XML strings.

        Returns:
            Combined <sigml> XML string.
        """
        if not sigml_strings:
            return "<sigml></sigml>"

        combined_hns_signs: List[str] = []

        for idx, content in enumerate(sigml_strings):
            if not content or not content.strip():
                continue
            try:
                root = ET.fromstring(content.strip())
                if root.tag.lower() == "sigml":
                    for child in root:
                        combined_hns_signs.append(ET.tostring(child, encoding="unicode"))
                elif root.tag.lower() == "hns_sign":
                    combined_hns_signs.append(ET.tostring(root, encoding="unicode"))
            except Exception as e:
                logger.warning(f"Skipping invalid SiGML fragment at index {idx}: {e}")

        inner_xml = "\n".join(combined_hns_signs)
        return f"<sigml>\n{inner_xml}\n</sigml>"
