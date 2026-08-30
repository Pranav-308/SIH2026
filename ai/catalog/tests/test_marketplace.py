"""
Unit and integration tests for Buyer Marketplace & Artisan Matching Engine & API.
"""

import pytest
from fastapi.testclient import TestClient

from api import app
from schema import MarketplaceSearchQuery, MarketplaceSearchResponse
from marketplace import MarketplaceEngine
from exceptions import InvalidMarketplaceQueryError

client = TestClient(app)


def test_marketplace_health_endpoint():
    response = client.get("/api/v1/marketplace/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Marketplace" in data["service"]


def test_empty_query():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery())
    assert isinstance(res, MarketplaceSearchResponse)
    assert res.success is True
    assert res.total_results > 0
    assert len(res.results) > 0


def test_keyword_search():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(query="terracotta vase"))
    assert res.success is True
    assert res.total_results > 0
    top = res.results[0]
    assert "terracotta" in top.product.product_name.lower() or "vase" in top.product.product_name.lower()
    assert any("Keyword matched" in r for r in top.match_reasons)


def test_category_search():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(category="Pottery"))
    assert res.success is True
    assert res.total_results > 0
    for result in res.results:
        assert result.product.category.lower() == "pottery"


def test_craft_type_search():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(craft_type="Channapatna Toy"))
    assert res.success is True
    assert res.total_results > 0
    assert res.results[0].product.craft_type == "Channapatna Toy"


def test_location_filtering():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(location="Karnataka"))
    assert res.success is True
    assert res.total_results > 0
    for result in res.results:
        assert "karnataka" in result.product.artisan.location.lower()


def test_price_range_filtering():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(price_min=500.0, price_max=1000.0))
    assert res.success is True
    assert res.total_results > 0
    for result in res.results:
        assert 500.0 <= result.product.price_inr <= 1000.0


def test_combined_filters():
    engine = MarketplaceEngine()
    query = MarketplaceSearchQuery(
        query="terracotta vase",
        category="Pottery",
        location="Karnataka",
        price_min=300.0,
        price_max=1500.0,
    )
    res = engine.search_products(query)
    assert res.success is True
    assert res.total_results > 0
    top = res.results[0]
    assert top.product.category.lower() == "pottery"
    assert "karnataka" in top.product.artisan.location.lower()
    assert top.relevance_score >= 0.70


def test_ranking():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(query="blue pottery"))
    assert res.success is True
    scores = [r.relevance_score for r in res.results]
    assert scores == sorted(scores, reverse=True)


def test_no_matching_products():
    engine = MarketplaceEngine()
    res = engine.search_products(MarketplaceSearchQuery(query="nonexistent_xyz_item", category="NonExistentCategory"))
    assert res.success is True
    assert res.total_results == 0
    assert len(res.results) == 0


from pydantic import ValidationError


def test_invalid_input():
    engine = MarketplaceEngine()

    with pytest.raises(ValidationError):
        MarketplaceSearchQuery(price_min=-100.0)

    with pytest.raises(InvalidMarketplaceQueryError):
        engine.validate_query(MarketplaceSearchQuery(price_min=1000.0, price_max=500.0))



def test_api_marketplace_search_endpoint():
    payload = {
        "query": "terracotta vase",
        "category": "Pottery",
        "location": "Karnataka",
        "price_min": 300.0,
        "price_max": 1500.0,
    }
    response = client.post("/api/v1/marketplace/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_results"] > 0
    results = data["results"]
    assert len(results) > 0
    assert "product" in results[0]
    assert "artisan" in results[0]["product"]
    assert "relevance_score" in results[0]
    assert "match_reasons" in results[0]
