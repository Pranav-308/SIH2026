"""
Multilingual Speech-to-Text (STT) Module for Artisan Voice Input.
Provides pluggable STT architecture supporting Indian languages (Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Bengali, English).
"""

from abc import ABC, abstractmethod
import json
import logging
from typing import Optional, List
from pydantic import BaseModel, Field

from config import Config
from exceptions import (
    MissingAudioError,
    InvalidAudioError,
    UnsupportedAudioFormatError,
    EmptyTranscriptionError,
    MissingAPIKeyError,
    AIServiceError,
    InvalidAIResponseError,
)
from schema import VoiceTranscriptionResponse

logger = logging.getLogger("voice_ai_service")

# Supported Audio MIME types & extensions
SUPPORTED_MIME_TYPES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
}

# Domain-specific artisan craft vocabulary hints to improve transcription accuracy
DEFAULT_ARTISAN_VOCABULARY: List[str] = [
    "Dhokra", "Bidriware", "Phulkari", "Pattachitra", "Madhubani", "Warli",
    "Terracotta", "Bandhani", "Kantha", "Bamboo", "Handloom", "Handicraft",
    "Channapatna", "Blue Pottery", "Chikankari", "Block Print", "Pashmina",
    "Kanjeevaram", "Zardozi", "Brassware", "Woodcarving", "Tanjore"
]

STT_PROMPT_TEMPLATE = """You are an expert multilingual AI Speech-to-Text transcription engine specializing in Indian languages (Kannada, Hindi, Tamil, Telugu, Malayalam, Marathi, Bengali, English) and artisan craft terms.

Audio Domain Context & Vocabulary Hints:
{vocabulary_hints}

Language Hint: {language_hint}

Instructions:
1. Listen carefully to the provided audio file.
2. Transcribe the spoken text accurately in its original language (or code-switched script if spoken in mixed Hinglish/Kanglish).
3. Identify the spoken language and return its ISO code or name (e.g. kn-IN for Kannada, hi-IN for Hindi, en-US for English, ta-IN for Tamil, te-IN for Telugu, ml-IN for Malayalam, mr-IN for Marathi, bn-IN for Bengali).
4. Preserve domain craft terminology accurately (e.g. Terracotta, Kantha, Bandhani, Bamboo).
5. Output ONLY valid JSON matching this structure:
{{
  "transcribed_text": "<transcribed text here>",
  "detected_language": "<language code/name>",
  "confidence": <float between 0.0 and 1.0 or null>
}}
"""


class BaseSpeechToTextEngine(ABC):
    """Abstract Base Class for pluggable Speech-to-Text engines."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_hint: Optional[str] = None,
        custom_vocabulary: Optional[List[str]] = None
    ) -> VoiceTranscriptionResponse:
        """
        Transcribes audio bytes into text and detects spoken language.

        :param audio_bytes: Raw bytes of the audio file.
        :param mime_type: MIME type string (e.g. audio/wav, audio/mp3).
        :param language_hint: Optional hint for expected language (e.g. 'kn-IN', 'hi-IN').
        :param custom_vocabulary: Optional list of craft domain terms to boost accuracy.
        :return: VoiceTranscriptionResponse instance.
        """
        pass


class GeminiTranscriptionEngine(BaseSpeechToTextEngine):
    """
    Primary Multilingual Speech-to-Text Engine powered by Gemini 3.6 Flash.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._api_key = api_key
        self._model_name = model_name or Config.get_model_name()

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return Config.get_api_key(strict=True)

    def validate_audio(self, audio_bytes: bytes, mime_type: str) -> str:
        """Validates audio file non-emptiness and MIME type support."""
        if not audio_bytes or len(audio_bytes) == 0:
            raise MissingAudioError("Audio file bytes cannot be empty.")

        clean_mime = mime_type.strip().lower() if mime_type else "audio/wav"
        if clean_mime not in SUPPORTED_MIME_TYPES:
            # Check if mime_type contains acceptable substring
            matched = False
            for supported_mime in SUPPORTED_MIME_TYPES:
                if supported_mime in clean_mime:
                    clean_mime = supported_mime
                    matched = True
                    break
            if not matched:
                raise UnsupportedAudioFormatError(
                    f"Unsupported audio format '{mime_type}'. Supported formats: WAV, MP3, M4A, OGG, FLAC, AAC."
                )

        return clean_mime

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_hint: Optional[str] = None,
        custom_vocabulary: Optional[List[str]] = None
    ) -> VoiceTranscriptionResponse:
        clean_mime = self.validate_audio(audio_bytes, mime_type)
        api_key = self._get_api_key()

        # Combine vocabulary hints
        vocab_list = list(DEFAULT_ARTISAN_VOCABULARY)
        if custom_vocabulary:
            vocab_list.extend(custom_vocabulary)
        vocab_str = ", ".join(list(dict.fromkeys(vocab_list)))

        lang_hint_str = language_hint.strip() if language_hint and language_hint.strip() else "Automatic Detection"
        prompt = STT_PROMPT_TEMPLATE.format(
            vocabulary_hints=vocab_str,
            language_hint=lang_hint_str
        )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # Create Inline Data Part for Audio
            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type=clean_mime
            )

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )

            logger.info(f"Executing speech recognition using model: '{self._model_name}'")
            response = client.models.generate_content(
                model=self._model_name,
                contents=[audio_part, prompt],
                config=config,
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise EmptyTranscriptionError("Audio contained no clear or audible speech to transcribe.")

        except (MissingAPIKeyError, MissingAudioError, UnsupportedAudioFormatError, EmptyTranscriptionError):
            raise
        except Exception as e:
            if "GEMINI_API_KEY" in str(e):
                raise MissingAPIKeyError(str(e))
            logger.error(f"Error during Gemini Speech-to-Text call with model '{self._model_name}': {str(e)}", exc_info=True)
            raise AIServiceError(f"Speech recognition model '{self._model_name}' failed or is unavailable: {str(e)}")

        # Parse output JSON into VoiceTranscriptionResponse
        try:
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            parsed = json.loads(cleaned_text)
            text = parsed.get("transcribed_text", "").strip()
            if not text:
                raise EmptyTranscriptionError("Audio transcription output was empty.")

            detected_lang = parsed.get("detected_language", "Unknown").strip()
            conf = parsed.get("confidence")
            conf_val = float(conf) if conf is not None else None

            return VoiceTranscriptionResponse(
                transcribed_text=text,
                detected_language=detected_lang,
                confidence=conf_val
            )

        except EmptyTranscriptionError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse STT response: {raw_text}", exc_info=True)
            raise InvalidAIResponseError(f"Failed to parse Speech-to-Text response JSON: {str(e)}")


class FasterWhisperEngine(BaseSpeechToTextEngine):
    """
    Future 100% offline Speech-to-Text Engine fallback using CTranslate2 faster-whisper.
    Currently reserved for future edge deployment.
    """

    def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_hint: Optional[str] = None,
        custom_vocabulary: Optional[List[str]] = None
    ) -> VoiceTranscriptionResponse:
        raise NotImplementedError("FasterWhisperEngine offline engine is reserved for future edge deployment.")


class VoiceService:
    """High-level service manager for artisan voice recognition."""

    def __init__(self, engine: Optional[BaseSpeechToTextEngine] = None):
        self.engine = engine or GeminiTranscriptionEngine()

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_hint: Optional[str] = None,
        custom_vocabulary: Optional[List[str]] = None
    ) -> VoiceTranscriptionResponse:
        return self.engine.transcribe(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            language_hint=language_hint,
            custom_vocabulary=custom_vocabulary
        )
