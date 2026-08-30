"""
Unit and integration tests for Smart Pricing Intelligence AI Engine & API.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api import app
from schema import ProductCatalog, PricingInput, PricingEstimateResponse
from pricing import PricingEngine
from exceptions import MissingPricingInputError, InvalidPricingInputError, PricingDatasetError

client = TestClient(app)


@pytest.fixture
def sample_catalog():
    return ProductCatalog(
        product_name="Handcrafted Geometric Terracotta Clay Vase",
        category="Pottery",
        description="An exquisitely handcrafted terracotta clay vase with geometric silhouette.",
        materials=["Clay", "Terracotta"],
        craft_type="Terracotta",
        colors=["Terracotta", "Rust"],
        tags=["Terracotta Vase", "Pottery", "Clay Decor"],
        confidence_score=0.95,
    )


@pytest.fixture
def sample_pricing_input(sample_catalog):
    return PricingInput(
        catalog=sample_catalog,
        material_cost_inr=150.0,
        labor_hours=6.0,
        artisan_hourly_wage_inr=100.0,
        packaging_shipping_cost_inr=50.0,
        market_tier="Fair Trade / Artisanal",
    )


def test_cost_breakdown_calculation(sample_pricing_input):
    engine = PricingEngine()
    cb = engine.calculate_cost_breakdown(sample_pricing_input)

    # Material: 150, Labor: 6 * 100 = 600, Overhead: 50
    # Base: 150 + 600 + 50 = 800
    # Artisan Margin (15%): 800 * 0.15 = 120
    # Cost Floor: 800 + 120 = 920
    assert cb.material_cost_inr == 150.0
    assert cb.labor_cost_inr == 600.0
    assert cb.overhead_inr == 50.0
    assert cb.artisan_margin_inr == 120.0
    assert cb.cost_floor_inr == 920.0


from pydantic import ValidationError


def test_invalid_input_validation(sample_catalog):
    engine = PricingEngine()

    with pytest.raises(ValidationError):
        PricingInput(catalog=sample_catalog, material_cost_inr=-10.0)

    with pytest.raises(ValidationError):
        PricingInput(catalog=sample_catalog, labor_hours=-2.0)

    with pytest.raises(ValidationError):
        PricingInput(catalog=sample_catalog, artisan_hourly_wage_inr=-50.0)

    with pytest.raises(MissingPricingInputError):
        engine.validate_input(None)



def test_dataset_loading():
    engine = PricingEngine()
    dataset = engine.load_dataset()
    assert isinstance(dataset, list)
    assert len(dataset) > 0
    assert "craft_type" in dataset[0]


def test_similarity_matching(sample_catalog):
    engine = PricingEngine()
    comps = engine.find_comparable_products(sample_catalog, top_k=3)
    assert len(comps) > 0
    # Terracotta craft items should rank highest
    top_craft = comps[0].get("craft_type")
    assert "Terracotta" in top_craft or comps[0]["similarity_score"] > 0


def test_confidence_score_bounds(sample_catalog):
    engine = PricingEngine()
    comps = engine.find_comparable_products(sample_catalog, top_k=5)
    conf = engine.calculate_confidence_score(0.95, comps)
    assert 0.0 <= conf <= 1.0


@patch("google.genai.Client")
def test_pricing_engine_mocked(mock_client_cls, sample_pricing_input):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '''```json
    {
        "recommended_price_inr": 1150.0,
        "min_price_inr": 950.0,
        "max_price_inr": 1400.0,
        "pricing_explanation": "Fair price for terracotta vase with artisan margin."
    }
    ```'''
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    engine = PricingEngine(api_key="mock_key")
    res = engine.estimate_price(sample_pricing_input)

    assert isinstance(res, PricingEstimateResponse)
    assert res.recommended_price_inr == 1150.0
    assert res.price_range.min_price_inr == 950.0
    assert res.price_range.max_price_inr == 1400.0
    assert res.cost_breakdown.cost_floor_inr == 920.0
    assert res.confidence_score > 0.0


@patch("google.genai.Client")
def test_api_pricing_estimate_endpoint(mock_client_cls, sample_pricing_input):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '''```json
    {
        "recommended_price_inr": 1150.0,
        "min_price_inr": 950.0,
        "max_price_inr": 1400.0,
        "pricing_explanation": "Fair price for terracotta vase with artisan margin."
    }
    ```'''
    mock_client.models.generate_content.return_value = mock_response
    mock_client_cls.return_value = mock_client

    payload = sample_pricing_input.model_dump()

    response = client.post("/api/v1/pricing/estimate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_price_inr" in data
    assert data["recommended_price_inr"] == 1150.0
    assert "cost_breakdown" in data
    assert data["cost_breakdown"]["cost_floor_inr"] == 920.0
