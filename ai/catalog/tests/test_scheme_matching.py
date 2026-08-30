"""
Unit and integration tests for Government Scheme Matching AI Engine & API.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from api import app
from schema import SchemeMatchingRequest, SchemeMatchingResponse
from scheme_matching import SchemeMatchingEngine
from exceptions import InvalidSchemeRequestError, PricingDatasetError

client = TestClient(app)


def test_schemes_health_endpoint():
    response = client.get("/api/v1/schemes/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Government Scheme" in data["service"]


def test_basic_scheme_matching():
    engine = SchemeMatchingEngine()
    req = SchemeMatchingRequest(
        craft_type="Terracotta",
        location="Karnataka",
        business_status="Individual Artisan",
        registration_status="Unregistered / Informal",
        age=30,
        gender="Female",
    )
    res = engine.match_schemes(req)
    assert isinstance(res, SchemeMatchingResponse)
    assert res.success is True
    assert res.total_matched >= 3
    top_scheme = res.results[0].scheme
    assert "PM Vishwakarma" in top_scheme.scheme_name or "Karnataka" in top_scheme.scheme_name
    assert top_scheme.official_source_url.startswith("http")


def test_state_specific_filtering():
    engine = SchemeMatchingEngine()
    req = SchemeMatchingRequest(
        craft_type="Blue Pottery",
        location="Rajasthan",
        business_status="Individual Artisan",
    )
    res = engine.match_schemes(req)
    matched_ids = [r.scheme.scheme_id for r in res.results]
    # Karnataka state scheme should NOT match a Rajasthan artisan
    assert "SCHEME-KARNATAKA-ARTISAN" not in matched_ids
    assert "SCHEME-PM-VISHWAKARMA" in matched_ids or "SCHEME-PEHCHAN-CARD" in matched_ids


def test_female_only_demographic_rule():
    engine = SchemeMatchingEngine()
    
    # Male artisan request -> Mahila Coir Yojana should NOT match
    male_req = SchemeMatchingRequest(
        craft_type="Natural Fiber Craft",
        location="Assam",
        gender="Male"
    )
    male_res = engine.match_schemes(male_req)
    male_matched_ids = [r.scheme.scheme_id for r in male_res.results]
    assert "SCHEME-MAHILA-COIR" not in male_matched_ids

    # Female artisan request -> Mahila Coir Yojana SHOULD match
    female_req = SchemeMatchingRequest(
        craft_type="Natural Fiber Craft",
        location="Assam",
        gender="Female"
    )
    female_res = engine.match_schemes(female_req)
    female_matched_ids = [r.scheme.scheme_id for r in female_res.results]
    assert "SCHEME-MAHILA-COIR" in female_matched_ids


def test_underage_demographic_rule():
    engine = SchemeMatchingEngine()
    req = SchemeMatchingRequest(
        craft_type="Terracotta",
        location="Karnataka",
        age=15  # Underage (<18)
    )
    res = engine.match_schemes(req)
    matched_ids = [r.scheme.scheme_id for r in res.results]
    # PM Vishwakarma requires age >= 18
    assert "SCHEME-PM-VISHWAKARMA" not in matched_ids


def test_invalid_request_validation():
    engine = SchemeMatchingEngine()

    with pytest.raises(InvalidSchemeRequestError):
        engine.validate_request(SchemeMatchingRequest(craft_type="", location="Karnataka"))

    with pytest.raises(InvalidSchemeRequestError):
        engine.validate_request(SchemeMatchingRequest(craft_type="Pottery", location="  "))


def test_missing_dataset_file():
    engine = SchemeMatchingEngine(dataset_path=Path("non_existent_schemes.json"))
    with pytest.raises(PricingDatasetError):
        engine.load_schemes_dataset()


def test_api_scheme_matching_endpoint():
    payload = {
        "craft_type": "Terracotta",
        "location": "Karnataka",
        "business_status": "Individual Artisan",
        "registration_status": "Unregistered / Informal",
        "age": 32,
        "gender": "Female"
    }
    response = client.post("/api/v1/schemes/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["total_matched"] > 0
    results = data["results"]
    assert "scheme" in results[0]
    assert "match_score" in results[0]
    assert "matched_criteria" in results[0]
    assert "official_source_url" in results[0]["scheme"]
    assert results[0]["scheme"]["official_source_url"].startswith("http")
