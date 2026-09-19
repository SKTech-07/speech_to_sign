import os
import tempfile
import subprocess
import logging
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

MIN_AUDIO_BYTES = 2000

def validate_audio_bytes(content: bytes) -> None:
    """
    Validates that the uploaded audio file meets minimum byte count.
    Rejects under-sized uploads with HTTP 422.
    """
    byte_count = len(content)
    logger.info(f"Received audio upload size: {byte_count} bytes")
    
    if byte_count < MIN_AUDIO_BYTES:
        logger.warning(f"Audio upload rejected: {byte_count} bytes is under minimum {MIN_AUDIO_BYTES} bytes")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "AUDIO_DECODE_FAILED",
                "message": "We couldn't read that recording. Please try again."
            }
        )

def transcode_to_wav(input_path: str) -> str:
    """
    Explicitly transcodes any input audio file to 16kHz mono 16-bit PCM WAV using ffmpeg.
    
    Returns the path to the temporary WAV file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
    # Log duration/codec metadata via ffprobe if available
    _probe_audio_metadata(input_path)

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)  # Close open file handle immediately on Windows to prevent ffmpeg file lock

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        wav_path
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            logger.error(f"FFmpeg transcoding failed (exit code {proc.returncode}): {proc.stderr}")
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "AUDIO_DECODE_FAILED",
                    "message": "We couldn't read that recording. Please try again."
                }
            )
        return wav_path

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg transcoding timed out after 30 seconds")
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "AUDIO_DECODE_FAILED",
                "message": "We couldn't read that recording. Please try again."
            }
        )

def _probe_audio_metadata(input_path: str) -> None:
    """
    Runs ffprobe to log input audio duration and codec at INFO level.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if proc.returncode == 0 and proc.stdout:
            lines = proc.stdout.strip().splitlines()
            logger.info(f"Audio metadata probe for '{os.path.basename(input_path)}': {lines}")
    except Exception as e:
        logger.debug(f"ffprobe log ignored: {e}")
