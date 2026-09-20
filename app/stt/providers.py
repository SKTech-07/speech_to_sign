import os
import time
import wave
import tempfile
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel

from app.stt.post_processor import build_initial_prompt, correct_transcript

logger = logging.getLogger(__name__)


class TranscriptResult(BaseModel):
    text: str
    raw_text: str
    latency_seconds: float
    status: str  # "SUCCESS", "NO_SPEECH_DETECTED", "AUDIO_DECODE_FAILED", "STT_UNAVAILABLE"
    error_message: Optional[str] = None


class BaseSTTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptResult:
        pass

    @abstractmethod
    def warmup(self) -> None:
        pass


class LocalWhisperProvider(BaseSTTProvider):
    """
    Local CTranslate2 (faster-whisper) Speech-to-Text provider.
    """
    def __init__(self, model_name: Optional[str] = None):
        import torch
        from faster_whisper import WhisperModel

        self.model_name = model_name or os.getenv("STT_MODEL", "small.en")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        cpu_threads = min(4, os.cpu_count() or 4)

        logger.info(
            f"Initializing LocalWhisperProvider (faster-whisper) with model='{self.model_name}', "
            f"device='{device}', compute_type='{compute_type}', cpu_threads={cpu_threads}"
        )

        try:
            self.model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads
            )
            self.initial_prompt = build_initial_prompt()
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model '{self.model_name}': {e}")
            raise RuntimeError(f"Could not load local STT model '{self.model_name}': {e}")

    def warmup(self) -> None:
        """
        Warms up the model by transcribing 1 second of generated silence.
        """
        logger.info("Warming up LocalWhisperProvider model on 1 second silent WAV buffer...")
        start_time = time.perf_counter()
        
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00" * 32000)  # 1 sec silence
                
            segments, _ = self.model.transcribe(
                temp_wav,
                language="en",
                task="transcribe",
                beam_size=1,
                condition_on_previous_text=False
            )
            list(segments)
            elapsed = time.perf_counter() - start_time
            logger.info(f"LocalWhisperProvider warmup completed in {elapsed:.3f}s")
        except Exception as e:
            logger.warning(f"STT warmup encounter: {e}")
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def transcribe(self, audio_path: str) -> TranscriptResult:
        start_time = time.perf_counter()
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                beam_size=5,
                temperature=[0.0, 0.2, 0.4],
                condition_on_previous_text=False,
                vad_filter=False,
                initial_prompt=self.initial_prompt
            )

            raw_text_parts = [segment.text for segment in segments]
            raw_text = " ".join(raw_text_parts).strip()
            elapsed = time.perf_counter() - start_time

            corrected_text, is_speech = correct_transcript(raw_text)

            logger.info(
                f"STT Local Completed in {elapsed:.3f}s | Raw: '{raw_text}' | Corrected: '{corrected_text}'"
            )

            if not is_speech or not corrected_text:
                return TranscriptResult(
                    text="",
                    raw_text=raw_text,
                    latency_seconds=elapsed,
                    status="NO_SPEECH_DETECTED"
                )

            return TranscriptResult(
                text=corrected_text,
                raw_text=raw_text,
                latency_seconds=elapsed,
                status="SUCCESS"
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"LocalWhisperProvider transcription error ({elapsed:.3f}s): {e}")
            return TranscriptResult(
                text="",
                raw_text="",
                latency_seconds=elapsed,
                status="STT_UNAVAILABLE",
                error_message="Voice input is temporarily unavailable."
            )


class OpenAIWhisperProvider(BaseSTTProvider):
    """
    OpenAI Cloud Speech-to-Text provider with automatic fallback to LocalWhisperProvider.
    """
    def __init__(self, fallback_provider: Optional[BaseSTTProvider] = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("STT_API_MODEL", "whisper-1")
        self.fallback_provider = fallback_provider or LocalWhisperProvider()
        self.initial_prompt = build_initial_prompt()

        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set. OpenAIWhisperProvider will fall back to LocalWhisperProvider.")

    def warmup(self) -> None:
        self.fallback_provider.warmup()

    def transcribe(self, audio_path: str) -> TranscriptResult:
        if not self.api_key:
            logger.info("No OPENAI_API_KEY available. Falling back to local STT provider.")
            return self.fallback_provider.transcribe(audio_path)

        start_time = time.perf_counter()
        import openai

        client = openai.OpenAI(api_key=self.api_key, timeout=15.0, max_retries=1)

        try:
            with open(audio_path, "rb") as audio_file:
                res = client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                    language="en",
                    prompt=self.initial_prompt
                )
            raw_text = res.text.strip() if hasattr(res, "text") else str(res).strip()
            elapsed = time.perf_counter() - start_time

            corrected_text, is_speech = correct_transcript(raw_text)
            logger.info(
                f"STT OpenAI Cloud Completed in {elapsed:.3f}s | Raw: '{raw_text}' | Corrected: '{corrected_text}'"
            )

            if not is_speech or not corrected_text:
                return TranscriptResult(
                    text="",
                    raw_text=raw_text,
                    latency_seconds=elapsed,
                    status="NO_SPEECH_DETECTED"
                )

            return TranscriptResult(
                text=corrected_text,
                raw_text=raw_text,
                latency_seconds=elapsed,
                status="SUCCESS"
            )

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.warning(f"OpenAI Cloud STT failed ({e}). Falling back to local provider.")
            return self.fallback_provider.transcribe(audio_path)
