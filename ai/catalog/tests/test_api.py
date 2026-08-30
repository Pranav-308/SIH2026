import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from api import app
from service import CatalogAIService
from schema import ProductCatalog
from exceptions import MissingImageError, InvalidImageError, MissingAPIKeyError, AIServiceError

client = TestClient(app)


def create_sample_image_file():
    img = Image.new("RGB", (30, 30), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return ("sample.jpg", buf, "image/jpeg")


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@patch.object(CatalogAIService, "generate_catalog")
def test_generate_catalog_success(mock_generate_catalog):
    mock_catalog = ProductCatalog(
        product_name="Wooden Elephant Idol",
        category="Woodcraft",
        description="Carved teak wood elephant sculpture.",
        materials=["Teak Wood"],
        craft_type="Woodcarving",
        colors=["Brown"],
        tags=["wood", "elephant", "statue"],
        confidence_score=0.96
    )
    mock_generate_catalog.return_value = mock_catalog

    file_tuple = create_sample_image_file()
    response = client.post(
        "/api/v1/catalog/generate",
        files={"image": file_tuple},
        data={"artisan_description": "Hand-carved wooden elephant"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["product_name"] == "Wooden Elephant Idol"
    assert data["category"] == "Woodcraft"
    assert data["confidence_score"] == 0.96


def test_generate_catalog_missing_image():
    response = client.post("/api/v1/catalog/generate")
    assert response.status_code == 422  # FastAPI validation error for missing required parameter


@patch.object(CatalogAIService, "generate_catalog")
def test_generate_catalog_invalid_image_domain_error(mock_generate_catalog):
    mock_generate_catalog.side_effect = InvalidImageError("Invalid image format.")

    file_tuple = ("test.txt", io.BytesIO(b"not an image"), "text/plain")
    response = client.post(
        "/api/v1/catalog/generate",
        files={"image": file_tuple}
    )

    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Invalid image format" in data["error"]


@patch.object(CatalogAIService, "generate_catalog")
def test_generate_catalog_missing_api_key(mock_generate_catalog):
    mock_generate_catalog.side_effect = MissingAPIKeyError("GEMINI_API_KEY missing.")

    file_tuple = create_sample_image_file()
    response = client.post(
        "/api/v1/catalog/generate",
        files={"image": file_tuple}
    )

    assert response.status_code == 401
    data = response.json()
    assert data["success"] is False
    assert "GEMINI_API_KEY missing" in data["error"]
