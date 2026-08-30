import pytest
from pydantic import ValidationError
from schema import ProductCatalog, CatalogInput, ErrorResponse


def test_product_catalog_valid():
    catalog = ProductCatalog(
        product_name="Handcrafted Brass Diya",
        category="Home Decor",
        description="Traditional engraved brass lamp for rituals.",
        materials=["Brass"],
        craft_type="Brassware",
        colors=["Gold", "Yellow"],
        tags=["brass", "diya", "lamp"],
        confidence_score=0.92
    )
    assert catalog.product_name == "Handcrafted Brass Diya"
    assert catalog.confidence_score == 0.92  # 2 decimals
    assert "Brass" in catalog.materials


def test_product_catalog_confidence_score_out_of_bounds():
    with pytest.raises(ValidationError):
        ProductCatalog(
            product_name="Test Item",
            category="Test",
            description="Test desc",
            materials=[],
            craft_type="Test Craft",
            colors=[],
            tags=[],
            confidence_score=1.5  # Invalid score > 1.0
        )

    with pytest.raises(ValidationError):
        ProductCatalog(
            product_name="Test Item",
            category="Test",
            description="Test desc",
            materials=[],
            craft_type="Test Craft",
            colors=[],
            tags=[],
            confidence_score=-0.1  # Invalid score < 0.0
        )


def test_product_catalog_missing_required_fields():
    with pytest.raises(ValidationError):
        ProductCatalog(
            product_name="Test Item",
            # missing category, description, craft_type, confidence_score
        )


def test_catalog_input_schema():
    sample_bytes = b"fake_image_bytes"
    inp = CatalogInput(image_bytes=sample_bytes, artisan_description="Handmade by me")
    assert inp.image_bytes == sample_bytes
    assert inp.artisan_description == "Handmade by me"


def test_error_response_schema():
    err = ErrorResponse(success=False, error="Image invalid", status_code=400)
    assert err.success is False
    assert err.status_code == 400
