from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    text: str = Field(..., description="English text to translate to sign lookup terms", json_schema_extra={"example": "I want to drink water"})


class TranslateResponse(BaseModel):
    text: str = Field(..., description="Original English input text")
    gloss: List[str] = Field(..., description="Extracted sign lookup terms")


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="Transcribed (corrected) text from speech audio")
    raw_text: Optional[str] = Field(default=None, description="Raw uncorrected Whisper transcription text")
    status: Optional[str] = Field(default="SUCCESS", description="STT status code")


class SpeechToGlossResponse(BaseModel):
    text: str = Field(..., description="Transcribed English text")
    gloss: List[str] = Field(..., description="Extracted sign lookup terms")


class SignLookupRequest(BaseModel):
    text: str = Field(..., description="Word or term to search in the sign dictionary", json_schema_extra={"example": "afraid"})


class SignLookupResponse(BaseModel):
    found: bool = Field(..., description="Whether the sign exists in the dictionary")
    query: str = Field(..., description="Original query term")
    matched_key: Optional[str] = Field(default=None, description="Matched dictionary filename/key")
    source_file: Optional[str] = Field(default=None, description="Path to .sigml source file")
    format: Optional[str] = Field(default=None, description="Sign definition format (e.g. sigml)")
    status: str = Field(..., description="Result status e.g. READY_FOR_CWASA or SIGN_NOT_AVAILABLE")
    sigml: Optional[str] = Field(default=None, description="Actual validated SiGML XML content")
    error: Optional[str] = Field(default=None, description="Error details if invalid")


class GlossToSignRequest(BaseModel):
    gloss: List[str] = Field(..., description="List of sign lookup terms", json_schema_extra={"example": ["afraid"]})


class GlossToSignResponse(BaseModel):
    status: str = Field(..., description="Overall sequence status e.g. READY_FOR_CWASA")
    gloss: List[str] = Field(..., description="Received terms sequence")
    results: List[SignLookupResponse] = Field(default_factory=list, description="Per-term lookup results")
    combined_sigml: Optional[str] = Field(default=None, description="Combined SiGML XML sequence for CWASA player")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", json_schema_extra={"example": "ok"})


class SignSequenceToken(BaseModel):
    word: str = Field(..., description="Original input word token")
    gloss: str = Field(..., description="Matched sign gloss key")
    available: bool = Field(..., description="Whether a valid sign file was found")


class SignSequenceRequest(BaseModel):
    text: str = Field(..., description="English sentence to translate into a single combined SiGML sequence", json_schema_extra={"example": "I need water"})


class SignSequenceResponse(BaseModel):
    text: str = Field(..., description="Original input text")
    tokens: List[SignSequenceToken] = Field(..., description="Word tokens with availability status")
    sigml: str = Field(..., description="Combined SiGML XML document for CWASA player execution")

