import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from api import app
from service import CatalogAIService
from voice import VoiceService
from schema import ProductCatalog, VoiceTranscriptionResponse

client = TestClient(app)


def create_sample_image_file():
    img = Image.new("RGB", (30, 30), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return ("sample.jpg", buf, "image/jpeg")


def create_sample_audio_file():
    buf = io.BytesIO(b"RIFF_fake_wav_audio_content_data")
    return ("sample.wav", buf, "audio/wav")


@patch.object(VoiceService, "transcribe_audio")
@patch.object(CatalogAIService, "generate_catalog")
def test_combined_text_and_voice_catalog_generation(mock_catalog, mock_transcribe):
    mock_transcribe.return_value = VoiceTranscriptionResponse(
        transcribed_text="Handmade woven bamboo storage basket for home decor",
        detected_language="en-US",
        confidence=0.95
    )

    mock_catalog.return_value = ProductCatalog(
        product_name="Eco-friendly Bamboo Storage Basket",
        category="Home Decor & Storage",
        description="Beautiful bamboo storage basket woven by artisans.",
        materials=["Bamboo"],
        craft_type="Bamboo & Cane Craft",
        colors=["Beige"],
        tags=["bamboo", "basket", "storage"],
        confidence_score=0.96
    )

    img_tuple = create_sample_image_file()
    audio_tuple = create_sample_audio_file()

    response = client.post(
        "/api/v1/catalog/generate",
        files={
            "image": img_tuple,
            "artisan_voice": audio_tuple
        },
        data={
            "artisan_description": "Typed note: Made in Assam.",
            "language_hint": "en-US"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["product_name"] == "Eco-friendly Bamboo Storage Basket"
    assert data["craft_type"] == "Bamboo & Cane Craft"
    assert data["confidence_score"] == 0.96


@patch.object(CatalogAIService, "generate_catalog_combined")
def test_catalog_endpoint_voice_only(mock_combined):
    mock_combined.return_value = ProductCatalog(
        product_name="Hand-molded Terracotta Vase",
        category="Pottery",
        description="Terracotta vase painted with natural dyes.",
        materials=["Clay"],
        craft_type="Terracotta Pottery",
        colors=["Terracotta Red"],
        tags=["terracotta", "vase"],
        confidence_score=0.93
    )

    img_tuple = create_sample_image_file()
    audio_tuple = create_sample_audio_file()

    response = client.post(
        "/api/v1/catalog/generate",
        files={
            "image": img_tuple,
            "artisan_voice": audio_tuple
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["product_name"] == "Hand-molded Terracotta Vase"
    assert data["craft_type"] == "Terracotta Pottery"
