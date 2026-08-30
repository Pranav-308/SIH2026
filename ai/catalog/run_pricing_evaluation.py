"""
CLI evaluation tool for testing Smart Pricing Intelligence AI accuracy & financial metrics.

Usage:
  python run_pricing_evaluation.py
"""

import sys
import json
import logging
from pathlib import Path

# Ensure ai/catalog is in import path
sys.path.insert(0, str(Path(__file__).parent))

from schema import ProductCatalog, PricingInput
from pricing import PricingEngine
from exceptions import CatalogAIException

logger = logging.getLogger("pricing_evaluation")


def evaluate_pricing_dataset(eval_dataset_path: Path = None):
    eval_path = eval_dataset_path or (Path(__file__).parent / "pricing_evaluation_dataset.json")

    if not eval_path.exists():
        print(f"❌ Error: Evaluation dataset not found at '{eval_path}'")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("\n==================================================")
    print("💰 SMART PRICING INTELLIGENCE ENGINE EVALUATION")
    print("==================================================")
    print(f"Evaluation Dataset: {eval_path.name} ({len(eval_items)} test items)")
    print("--------------------------------------------------")

    engine = PricingEngine()
    
    ape_list = []
    floor_protected_count = 0
    in_range_count = 0
    total_count = len(eval_items)

    results = []

    for item in eval_items:
        eval_id = item.get("eval_id")
        cat_dict = item.get("catalog")
        catalog = ProductCatalog.model_validate(cat_dict)

        pricing_input = PricingInput(
            catalog=catalog,
            material_cost_inr=item.get("material_cost_inr", 0.0),
            labor_hours=item.get("labor_hours", 1.0),
            artisan_hourly_wage_inr=item.get("artisan_hourly_wage_inr", 100.0),
            packaging_shipping_cost_inr=item.get("packaging_shipping_cost_inr", 50.0),
            market_tier=item.get("market_tier", "Fair Trade / Artisanal"),
        )

        gt_actual = float(item.get("ground_truth_actual_price_inr", 0.0))
        gt_min = float(item.get("ground_truth_min_price_inr", 0.0))
        gt_max = float(item.get("ground_truth_max_price_inr", 0.0))

        try:
            res = engine.estimate_price(pricing_input)
            pred_price = res.recommended_price_inr
            cost_floor = res.cost_breakdown.cost_floor_inr

            # 1. Absolute Percentage Error
            ape = abs(gt_actual - pred_price) / gt_actual * 100.0 if gt_actual > 0 else 0.0
            ape_list.append(ape)

            # 2. Profit Floor Protection check
            if pred_price >= cost_floor:
                floor_protected_count += 1

            # 3. Ground Truth Range Capture check
            if gt_min <= pred_price <= gt_max:
                in_range_count += 1

            print(f"[{eval_id}] {catalog.product_name}")
            print(f"   Ground Truth Actual: ₹{gt_actual:.2f} | Recommended: ₹{pred_price:.2f} | Cost Floor: ₹{cost_floor:.2f}")
            print(f"   Error: {ape:.2f}% | Confidence: {res.confidence_score} | Comps Matched: {res.comparable_products_count}")
            print("--------------------------------------------------")

            results.append({
                "eval_id": eval_id,
                "product_name": catalog.product_name,
                "ground_truth_price": gt_actual,
                "recommended_price": pred_price,
                "cost_floor": cost_floor,
                "percentage_error": round(ape, 2),
                "floor_protected": pred_price >= cost_floor,
                "in_range": gt_min <= pred_price <= gt_max,
                "confidence_score": res.confidence_score,
            })

        except CatalogAIException as e:
            print(f"❌ Pricing AI Error on item [{eval_id}]: {e.message}")
        except Exception as e:
            print(f"❌ Unexpected Error on item [{eval_id}]: {str(e)}")

    mape = sum(ape_list) / len(ape_list) if ape_list else 0.0
    floor_protection_rate = (floor_protected_count / total_count * 100.0) if total_count > 0 else 0.0
    in_range_rate = (in_range_count / total_count * 100.0) if total_count > 0 else 0.0

    print("\n📊 EVALUATION SUMMARY:")
    print("==================================================")
    print(f"📈 Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"🛡️ Profit Floor Protection Rate:         {floor_protection_rate:.1f}%")
    print(f"🎯 In-Range Capture Rate:                 {in_range_rate:.1f}%")
    print("==================================================")

    output_file = Path(__file__).parent / "pricing_evaluation_summary.json"
    summary_data = {
        "mape_percent": round(mape, 2),
        "floor_protection_rate_percent": round(floor_protection_rate, 2),
        "in_range_capture_rate_percent": round(in_range_rate, 2),
        "total_test_items": total_count,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Full summary saved to: {output_file.name}")
    return summary_data


if __name__ == "__main__":
    evaluate_pricing_dataset()
