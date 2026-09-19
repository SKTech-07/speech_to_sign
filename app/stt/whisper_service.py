import os
import logging
from typing import Optional
from app.stt.providers import BaseSTTProvider, LocalWhisperProvider, OpenAIWhisperProvider, TranscriptResult

logger = logging.getLogger(__name__)

_stt_provider_instance: Optional[BaseSTTProvider] = None


def init_stt_provider() -> BaseSTTProvider:
    """
    Initializes and warms up the global STT provider singleton at application startup.
    """
    global _stt_provider_instance
    if _stt_provider_instance is not None:
        return _stt_provider_instance

    provider_type = os.getenv("STT_PROVIDER", "local").lower().strip()
    logger.info(f"Initializing STT Provider singleton (STT_PROVIDER='{provider_type}')...")

    if provider_type == "openai":
        local_fallback = LocalWhisperProvider()
        _stt_provider_instance = OpenAIWhisperProvider(fallback_provider=local_fallback)
    else:
        _stt_provider_instance = LocalWhisperProvider()

    # Perform warmup once at startup
    _stt_provider_instance.warmup()
    return _stt_provider_instance


def get_stt_provider() -> BaseSTTProvider:
    """
    Returns the initialized STT provider singleton.
    """
    global _stt_provider_instance
    if _stt_provider_instance is None:
        logger.warning("STT Provider requested before lifespan initialization. Initializing now...")
        _stt_provider_instance = init_stt_provider()
    return _stt_provider_instance


def transcribe_audio_sync(audio_path: str) -> TranscriptResult:
    """
    Synchronous transcription call to be executed off the main asyncio thread loop.
    """
    provider = get_stt_provider()
    return provider.transcribe(audio_path)


def transcribe_audio(audio_path: str) -> str:
    """
    Legacy compatibility wrapper returning text string.
    """
    res = transcribe_audio_sync(audio_path)
    return res.text
