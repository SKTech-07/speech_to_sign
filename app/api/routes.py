import os
import tempfile
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from app.models.schemas import (
    TranslateRequest, TranslateResponse,
    TranscribeResponse, SpeechToGlossResponse,
    SignLookupRequest, SignLookupResponse,
    GlossToSignRequest, GlossToSignResponse,
    SignSequenceRequest, SignSequenceResponse, SignSequenceToken,
    HealthResponse
)
from app.nlp.preprocess import preprocess_text
from app.nlp.analyzer import analyze_text
from app.gloss.gloss_generator import generate_gloss
from app.sign_dictionary.dictionary_service import SignDictionary
from app.cwasa.adapter import CWASAAdapter

logger = logging.getLogger(__name__)

router = APIRouter()
dictionary_service = SignDictionary()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint returning backend status.
    """
    return HealthResponse(status="ok")


@router.post("/sign/lookup", response_model=SignLookupResponse, tags=["Dictionary Lookup"])
async def lookup_single_sign(payload: SignLookupRequest):
    """
    Looks up a single sign in the generic dictionary and returns the actual validated .sigml content.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    res = dictionary_service.lookup_sign(payload.text)
    return SignLookupResponse(**res)


@router.post("/translate", response_model=TranslateResponse, tags=["Translation"])
async def translate_text(payload: TranslateRequest):
    """
    Processes English text into normalized sign lookup terms.
    """
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    cleaned_text = preprocess_text(payload.text)
    nlp_result = analyze_text(cleaned_text)
    terms = generate_gloss(cleaned_text, nlp_result)

    return TranslateResponse(
        text=payload.text,
        gloss=terms
    )


import asyncio
from app.stt.audio_utils import validate_audio_bytes, transcode_to_wav
from app.stt.whisper_service import transcribe_audio_sync

@router.post("/transcribe", response_model=TranscribeResponse, tags=["Speech-to-Text"])
async def transcribe_speech(file: UploadFile = File(...)):
    """
    Transcribes an uploaded speech audio file into text using faster-whisper.
    """
    content = await file.read()
    validate_audio_bytes(content)

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".webm"
    fd, raw_temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)

    wav_path = None
    try:
        wav_path = transcode_to_wav(raw_temp_path)
        result = await asyncio.to_thread(transcribe_audio_sync, wav_path)

        if result.status == "NO_SPEECH_DETECTED":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "NO_SPEECH_DETECTED",
                    "message": "We didn't hear any speech."
                }
            )
        elif result.status == "STT_UNAVAILABLE":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "STT_UNAVAILABLE",
                    "message": "Voice input is temporarily unavailable."
                }
            )
        elif result.status != "SUCCESS":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "AUDIO_DECODE_FAILED",
                    "message": "We couldn't read that recording. Please try again."
                }
            )

        return TranscribeResponse(
            text=result.text,
            raw_text=result.raw_text,
            status=result.status
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /transcribe: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "AUDIO_DECODE_FAILED",
                "message": "We couldn't read that recording. Please try again."
            }
        )
    finally:
        if os.path.exists(raw_temp_path):
            try:
                os.remove(raw_temp_path)
            except Exception:
                pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


@router.post("/speech-to-gloss", response_model=SpeechToGlossResponse, tags=["Full Pipeline"])
async def speech_to_gloss(file: UploadFile = File(...)):
    """
    Runs full Speech -> Text -> Text Processing -> Sign Lookup Terms pipeline.
    """
    content = await file.read()
    validate_audio_bytes(content)

    suffix = os.path.splitext(file.filename)[1] if file.filename else ".webm"
    fd, raw_temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)

    wav_path = None
    try:
        wav_path = transcode_to_wav(raw_temp_path)
        result = await asyncio.to_thread(transcribe_audio_sync, wav_path)

        if result.status != "SUCCESS" or not result.text:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": result.status if result.status != "SUCCESS" else "NO_SPEECH_DETECTED",
                    "message": "We didn't hear any speech."
                }
            )

        raw_text = result.text
        cleaned_text = preprocess_text(raw_text)
        nlp_result = analyze_text(cleaned_text)
        terms = generate_gloss(cleaned_text, nlp_result)

        return SpeechToGlossResponse(
            text=raw_text,
            gloss=terms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in speech-to-gloss pipeline: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "AUDIO_DECODE_FAILED",
                "message": "We couldn't read that recording. Please try again."
            }
        )
    finally:
        if os.path.exists(raw_temp_path):
            try:
                os.remove(raw_temp_path)
            except Exception:
                pass
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


@router.post("/gloss-to-sign", response_model=GlossToSignResponse, tags=["CWASA Boundary"])
async def gloss_to_sign(payload: GlossToSignRequest):
    """
    Looks up multiple terms in the sign dictionary, validates their SiGML, and produces
    CWASA-ready SiGML instructions.
    """
    if not payload.gloss:
        raise HTTPException(status_code=400, detail="Gloss list cannot be empty.")

    results = []
    valid_sigmls = []

    for term in payload.gloss:
        lookup_res = dictionary_service.lookup_sign(term)
        sign_model = SignLookupResponse(**lookup_res)
        results.append(sign_model)
        if sign_model.found and sign_model.sigml:
            valid_sigmls.append(sign_model.sigml)

    combined_sigml = CWASAAdapter.prepare_cwasa_sequence(valid_sigmls) if valid_sigmls else None
    overall_status = "READY_FOR_CWASA" if valid_sigmls else "SIGN_NOT_AVAILABLE"

    return GlossToSignResponse(
        status=overall_status,
        gloss=payload.gloss,
        results=results,
        combined_sigml=combined_sigml
    )


@router.post("/sign/sequence", response_model=SignSequenceResponse, tags=["Sign Sequence"])
async def create_sign_sequence(payload: SignSequenceRequest):
    """
    Tokenizes input text, matches words case-insensitively against data/SignFiles/,
    and returns a single combined SiGML XML document and token availability list.
    """
    import re
    if not payload.text or not payload.text.strip():
        return SignSequenceResponse(
            text=payload.text or "",
            tokens=[],
            sigml="<sigml></sigml>"
        )

    cleaned_input = payload.text.lower()
    raw_tokens = re.findall(r"\b\w+\b", cleaned_input)

    tokens_result = []
    matched_sigmls = []

    for word in raw_tokens:
        lookup_res = dictionary_service.lookup_sign(word)
        if lookup_res.get("found") and lookup_res.get("sigml"):
            matched_key = lookup_res.get("matched_key", word)
            tokens_result.append(SignSequenceToken(
                word=word,
                gloss=matched_key,
                available=True
            ))
            matched_sigmls.append(lookup_res["sigml"])
        else:
            tokens_result.append(SignSequenceToken(
                word=word,
                gloss=word,
                available=False
            ))

    combined_sigml = CWASAAdapter.prepare_cwasa_sequence(matched_sigmls)

    return SignSequenceResponse(
        text=payload.text,
        tokens=tokens_result,
        sigml=combined_sigml
    )
