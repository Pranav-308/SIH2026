import io
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from service import CatalogAIService
from schema import ProductCatalog
from exceptions import (
    MissingImageError,
    InvalidImageError,
    MissingAPIKeyError,
    AIServiceError,
    InvalidAIResponseError,
)


def create_sample_image_bytes() -> bytes:
    """Helper to generate valid JPEG image bytes in memory."""
    img = Image.new("RGB", (50, 50), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_image_bytes():
    return create_sample_image_bytes()


@pytest.fixture
def mock_catalog_json():
    return json.dumps({
        "product_name": "Handmade Blue Terracotta Pot",
        "category": "Pottery",
        "description": "Beautiful hand-molded clay pot painted with blue indigo dye.",
        "materials": ["Clay", "Indigo Dye"],
        "craft_type": "Terracotta",
        "colors": ["Blue", "Brown"],
        "tags": ["pottery", "terracotta", "handmade"],
        "confidence_score": 0.94
    })


def test_missing_image_bytes():
    service = CatalogAIService(api_key="mock_key")
    with pytest.raises(MissingImageError):
        service.generate_catalog(image_bytes=b"")


def test_invalid_image_bytes():
    service = CatalogAIService(api_key="mock_key")
    with pytest.raises(InvalidImageError):
        service.generate_catalog(image_bytes=b"invalid_non_image_bytes_string")


def test_missing_api_key(valid_image_bytes):
    with patch("os.getenv", return_value=None):
        service = CatalogAIService(api_key=None)
        with pytest.raises(MissingAPIKeyError):
            service.generate_catalog(image_bytes=valid_image_bytes)


@patch("google.genai.Client")
def test_successful_catalog_generation(mock_genai_client, valid_image_bytes, mock_catalog_json):
    # Mock Gemini response
    mock_response = MagicMock()
    mock_response.text = mock_catalog_json
    mock_genai_client.return_value.models.generate_content.return_value = mock_response

    service = CatalogAIService(api_key="mock_api_key")
    catalog = service.generate_catalog(
        image_bytes=valid_image_bytes,
        artisan_description="Terracotta pot made in Rajasthan"
    )

    assert isinstance(catalog, ProductCatalog)
    assert catalog.product_name == "Handmade Blue Terracotta Pot"
    assert catalog.craft_type == "Terracotta"
    assert catalog.confidence_score == 0.94


@patch("google.genai.Client")
def test_ai_service_api_failure(mock_genai_client, valid_image_bytes):
    mock_genai_client.return_value.models.generate_content.side_effect = Exception("API rate limit exceeded")

    service = CatalogAIService(api_key="mock_api_key")
    with pytest.raises(AIServiceError):
        service.generate_catalog(image_bytes=valid_image_bytes)


@patch("google.genai.Client")
def test_invalid_ai_response_json(mock_genai_client, valid_image_bytes):
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON"
    mock_genai_client.return_value.models.generate_content.return_value = mock_response

    service = CatalogAIService(api_key="mock_api_key")
    with pytest.raises(InvalidAIResponseError):
        service.generate_catalog(image_bytes=valid_image_bytes)
