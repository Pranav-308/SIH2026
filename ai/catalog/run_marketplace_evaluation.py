"""
CLI evaluation script for Buyer Marketplace Search & Recommendation Engine.

Usage:
  python run_marketplace_evaluation.py
"""

import sys
import json
import logging
from pathlib import Path

# Ensure ai/catalog is in import path
sys.path.insert(0, str(Path(__file__).parent))

from schema import MarketplaceSearchQuery
from marketplace import MarketplaceEngine
from exceptions import CatalogAIException

logger = logging.getLogger("marketplace_evaluation")


def evaluate_marketplace_dataset(eval_dataset_path: Path = None):
    eval_path = eval_dataset_path or (Path(__file__).parent / "marketplace_evaluation_dataset.json")

    if not eval_path.exists():
        print(f"❌ Error: Evaluation dataset not found at '{eval_path}'")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("\n==================================================")
    print("🛍️ BUYER MARKETPLACE RECOMMENDATION EVALUATION")
    print("==================================================")
    print(f"Evaluation Dataset: {eval_path.name} ({len(eval_items)} benchmark queries)")
    print("--------------------------------------------------")

    engine = MarketplaceEngine()

    top_1_hits = 0
    top_3_hits = 0
    cat_match_hits = 0
    total_queries = len(eval_items)

    results = []

    for item in eval_items:
        eval_id = item.get("eval_id")
        desc = item.get("description")
        sq_dict = item.get("search_query", {})
        search_query = MarketplaceSearchQuery.model_validate(sq_dict)

        expected_top_id = item.get("expected_top_product_id")
        expected_cat = item.get("expected_category")
        expected_top_3_ids = item.get("expected_top_3_product_ids", [])

        try:
            res = engine.search_products(search_query)
            top_results = res.results

            actual_top_id = top_results[0].product.product_id if top_results else None
            actual_top_cat = top_results[0].product.category if top_results else None
            actual_top_3_ids = [r.product.product_id for r in top_results[:3]]

            # Top-1 Accuracy check
            is_top_1 = (actual_top_id == expected_top_id)
            if is_top_1:
                top_1_hits += 1

            # Top-3 Accuracy check
            is_top_3 = (expected_top_id in actual_top_3_ids) or any(tid in expected_top_3_ids for tid in actual_top_3_ids)
            if is_top_3:
                top_3_hits += 1

            # Category match check
            is_cat_match = (actual_top_cat and expected_cat and actual_top_cat.lower() == expected_cat.lower())
            if is_cat_match:
                cat_match_hits += 1

            top_score = top_results[0].relevance_score if top_results else 0.0
            top_name = top_results[0].product.product_name if top_results else "None"

            print(f"[{eval_id}] {desc}")
            print(f"   Query: {sq_dict}")
            print(f"   Expected Top ID: {expected_top_id} | Actual Top ID: {actual_top_id} ({top_name})")
            print(f"   Top Score: {top_score} | Match Reasons: {top_results[0].match_reasons if top_results else []}")
            print("--------------------------------------------------")

            results.append({
                "eval_id": eval_id,
                "description": desc,
                "expected_top_id": expected_top_id,
                "actual_top_id": actual_top_id,
                "top_1_hit": is_top_1,
                "top_3_hit": is_top_3,
                "category_match_hit": is_cat_match,
                "top_score": top_score,
            })

        except CatalogAIException as e:
            print(f"❌ Marketplace Error on item [{eval_id}]: {e.message}")
        except Exception as e:
            print(f"❌ Unexpected Error on item [{eval_id}]: {str(e)}")

    top_1_accuracy = (top_1_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    top_3_accuracy = (top_3_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    cat_match_accuracy = (cat_match_hits / total_queries * 100.0) if total_queries > 0 else 0.0

    print("\n📊 EVALUATION SUMMARY (BENCHMARK SYNTHETIC SUITE):")
    print("==================================================")
    print(f"🎯 Top-1 Relevance Accuracy:        {top_1_accuracy:.1f}% ({top_1_hits}/{total_queries})")
    print(f"🎯 Top-3 Relevance Accuracy:        {top_3_accuracy:.1f}% ({top_3_hits}/{total_queries})")
    print(f"🏷️ Category Matching Accuracy:      {cat_match_accuracy:.1f}% ({cat_match_hits}/{total_queries})")
    print("==================================================")

    output_file = Path(__file__).parent / "marketplace_evaluation_summary.json"
    summary_data = {
        "top_1_relevance_accuracy_percent": round(top_1_accuracy, 2),
        "top_3_relevance_accuracy_percent": round(top_3_accuracy, 2),
        "category_matching_accuracy_percent": round(cat_match_accuracy, 2),
        "total_test_queries": total_queries,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Full summary saved to: {output_file.name}")
    return summary_data


if __name__ == "__main__":
    evaluate_marketplace_dataset()
