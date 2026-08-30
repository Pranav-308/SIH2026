import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app
from voice import (
    GeminiTranscriptionEngine,
    FasterWhisperEngine,
    VoiceService,
    SUPPORTED_MIME_TYPES
)
from schema import VoiceTranscriptionResponse
from exceptions import (
    MissingAudioError,
    InvalidAudioError,
    UnsupportedAudioFormatError,
    EmptyTranscriptionError,
    MissingAPIKeyError,
    AIServiceError,
    InvalidAIResponseError
)

client = TestClient(app)


def test_voice_transcription_response_confidence_default():
    # Confidence should default to None (NOT 1.0) to prevent false certainty
    resp = VoiceTranscriptionResponse(
        transcribed_text="Handmade bamboo basket",
        detected_language="en-US"
    )
    assert resp.confidence is None
    assert resp.transcribed_text == "Handmade bamboo basket"
    assert resp.detected_language == "en-US"


def test_validate_audio_empty_bytes():
    engine = GeminiTranscriptionEngine(api_key="mock_key")
    with pytest.raises(MissingAudioError):
        engine.validate_audio(b"", "audio/wav")


def test_validate_audio_unsupported_format():
    engine = GeminiTranscriptionEngine(api_key="mock_key")
    with pytest.raises(UnsupportedAudioFormatError):
        engine.validate_audio(b"fake_audio_bytes", "video/avi")


def test_faster_whisper_engine_unimplemented():
    engine = FasterWhisperEngine()
    with pytest.raises(NotImplementedError):
        engine.transcribe(b"audio", "audio/wav")


@patch("google.genai.Client")
def test_successful_gemini_transcription_english(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "transcribed_text": "This is a handmade bamboo basket crafted by rural artisans.",
        "detected_language": "en-US",
        "confidence": 0.96
    })
    mock_genai_client.return_value.models.generate_content.return_value = mock_response

    engine = GeminiTranscriptionEngine(api_key="mock_api_key")
    res = engine.transcribe(b"fake_audio_bytes", "audio/wav")

    assert isinstance(res, VoiceTranscriptionResponse)
    assert res.transcribed_text == "This is a handmade bamboo basket crafted by rural artisans."
    assert res.detected_language == "en-US"
    assert res.confidence == 0.96


@patch("google.genai.Client")
def test_successful_gemini_transcription_kannada(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "transcribed_text": "ಇದು ನೈಸರ್ಗಿಕ ಬಿದಿರಿನಿಂದ ಮಾಡಿದ ಕೈಯಿಂದ ಮಾಡಿದ ಬುಟ್ಟಿ",
        "detected_language": "kn-IN",
        "confidence": 0.94
    })
    mock_genai_client.return_value.models.generate_content.return_value = mock_response

    engine = GeminiTranscriptionEngine(api_key="mock_api_key")
    res = engine.transcribe(b"fake_audio_bytes", "audio/wav", language_hint="kn-IN")

    assert res.detected_language == "kn-IN"
    assert "ಬಿದಿರಿನಿಂದ" in res.transcribed_text


@patch("google.genai.Client")
def test_successful_gemini_transcription_hindi(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "transcribed_text": "यह प्राकृतिक मिट्टी से बना एक सुंदर टेराकोटा फूलदान है",
        "detected_language": "hi-IN",
        "confidence": None
    })
    mock_genai_client.return_value.models.generate_content.return_value = mock_response

    engine = GeminiTranscriptionEngine(api_key="mock_api_key")
    res = engine.transcribe(b"fake_audio_bytes", "audio/mp3", language_hint="hi-IN")

    assert res.detected_language == "hi-IN"
    assert res.confidence is None


def test_transcribe_api_endpoint_missing_audio():
    response = client.post("/api/v1/voice/transcribe")
    assert response.status_code == 422  # FastAPI missing parameter error


@patch.object(VoiceService, "transcribe_audio")
def test_transcribe_api_endpoint_success(mock_transcribe):
    mock_transcribe.return_value = VoiceTranscriptionResponse(
        transcribed_text="Traditional Bandhani silk saree with gold border",
        detected_language="en-IN",
        confidence=0.92
    )

    audio_file = ("sample.wav", io.BytesIO(b"fake_audio_data"), "audio/wav")
    response = client.post(
        "/api/v1/voice/transcribe",
        files={"audio": audio_file},
        data={"language_hint": "en-IN"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["transcribed_text"] == "Traditional Bandhani silk saree with gold border"
    assert data["detected_language"] == "en-IN"
    assert data["confidence"] == 0.92
