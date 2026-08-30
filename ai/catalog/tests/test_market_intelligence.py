"""
Unit and integration tests for Market Intelligence AI Engine & API.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api import app
from schema import MarketIntelligenceRequest, MarketIntelligenceResponse
from market_intelligence import MarketIntelligenceEngine
from exceptions import PricingDatasetError

client = TestClient(app)


def test_market_intelligence_health_endpoint():
    response = client.get("/api/v1/market-intelligence/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Market Intelligence" in data["service"]


def test_empty_request():
    engine = MarketIntelligenceEngine()
    res = engine.analyze(MarketIntelligenceRequest())
    assert isinstance(res, MarketIntelligenceResponse)
    assert res.success is True
    assert res.demand_level in ["Very High", "High", "Moderate", "Emerging"]
    assert res.market_opportunity in ["Strong", "Favorable", "Growing", "Niche"]
    assert len(res.trending_categories) > 0
    assert len(res.trending_crafts) > 0
    assert len(res.insights) > 0


def test_search_aggregation_and_location_demand():
    engine = MarketIntelligenceEngine()
    req = MarketIntelligenceRequest(location="Karnataka", craft_type="Terracotta", category="Pottery")
    res = engine.analyze(req)
    assert res.success is True
    assert res.demand_level in ["Very High", "High"]
    assert "Pottery" in res.trending_categories or "Terracotta" in res.trending_crafts
    assert any("Karnataka" in insight or "Terracotta" in insight for insight in res.insights)


def test_category_and_craft_trends():
    engine = MarketIntelligenceEngine()
    cats, crafts = engine.detect_trends()
    assert isinstance(cats, list)
    assert isinstance(crafts, list)
    assert len(cats) > 0
    assert len(crafts) > 0
    assert "Pottery" in cats or "Bamboo craft" in cats or "Wood craft" in cats


def test_price_trend_calculation():
    engine = MarketIntelligenceEngine()
    price_range = engine.calculate_optimal_price_range(craft_type="Terracotta", category="Pottery")
    assert isinstance(price_range, dict)
    assert "min_price" in price_range
    assert "median_price" in price_range
    assert "max_price" in price_range
    assert price_range["min_price"] <= price_range["median_price"] <= price_range["max_price"]


def test_artisan_specific_insights():
    engine = MarketIntelligenceEngine()
    req = MarketIntelligenceRequest(location="Assam", craft_type="Bamboo Weaving", category="Bamboo craft")
    res = engine.analyze(req)
    assert res.success is True
    assert len(res.insights) >= 3
    assert any("Bamboo" in ins for ins in res.insights)


def test_missing_dataset_handling():
    from pathlib import Path
    engine = MarketIntelligenceEngine(dataset_path=Path("non_existent_file.json"))
    with pytest.raises(PricingDatasetError):
        engine.load_activity_dataset()


def test_api_market_intelligence_analyze_endpoint():
    payload = {
        "location": "Karnataka",
        "craft_type": "Terracotta",
        "category": "Pottery"
    }
    response = client.post("/api/v1/market-intelligence/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "demand_level" in data
    assert "market_opportunity" in data
    assert "trending_categories" in data
    assert "trending_crafts" in data
    assert "optimal_price_range_inr" in data
    assert "insights" in data
    assert len(data["insights"]) > 0
