"""
CLI Tool for testing Smart Catalog AI end-to-end with real or sample product images.

Usage:
  python test_real_product.py --image path/to/image.jpg --description "Handmade bamboo basket"
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure ai/catalog is in import path
sys.path.insert(0, str(Path(__file__).parent))

from service import CatalogAIService
from exceptions import CatalogAIException


def run_catalog_test(image_path_str: str, description: str = ""):
    image_path = Path(image_path_str)
    if not image_path.exists():
        print(f"❌ Error: Image file not found at '{image_path}'")
        sys.exit(1)

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"❌ Error reading image file: {str(e)}")
        sys.exit(1)

    print("\n==================================================")
    print(f"📸 Image File: {image_path.name}")
    print(f"📝 Artisan Notes: {description if description else '(None)'}")
    print("==================================================")
    print("⏳ Processing through Gemini Multimodal Vision API...")

    try:
        service = CatalogAIService()
        catalog = service.generate_catalog(
            image_bytes=image_bytes,
            artisan_description=description
        )

        output_dict = catalog.model_dump()

        print("\n✅ SUCCESS: Structured Catalog Output Generated!")
        print("--------------------------------------------------")
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))
        print("--------------------------------------------------")

        # Field completeness check
        required_fields = [
            "product_name", "category", "description",
            "materials", "craft_type", "colors", "tags", "confidence_score"
        ]
        missing = [f for f in required_fields if f not in output_dict or output_dict[f] is None]
        if missing:
            print(f"⚠️ Warning: Missing or null fields: {missing}")
        else:
            print("🎉 All 8 required catalog fields populated successfully!")

        return output_dict

    except CatalogAIException as e:
        print(f"\n❌ Catalog AI Exception [{e.status_code}]: {e.message}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Failure: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Smart Catalog AI with image and artisan description.")
    parser.add_argument("--image", type=str, required=True, help="Path to product image file.")
    parser.add_argument("--description", type=str, default="", help="Optional artisan text description.")
    args = parser.parse_args()

    run_catalog_test(args.image, args.description)


if __name__ == "__main__":
    main()
