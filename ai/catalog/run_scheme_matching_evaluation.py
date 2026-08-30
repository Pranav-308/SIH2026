"""
CLI evaluation script for Government Scheme Matching Engine.

Usage:
  python run_scheme_matching_evaluation.py
"""

import sys
import json
import logging
from pathlib import Path

# Ensure ai/catalog is in import path
sys.path.insert(0, str(Path(__file__).parent))

from schema import SchemeMatchingRequest
from scheme_matching import SchemeMatchingEngine
from exceptions import CatalogAIException

logger = logging.getLogger("scheme_matching_evaluation")


def evaluate_scheme_matching_dataset(eval_dataset_path: Path = None):
    eval_path = eval_dataset_path or (Path(__file__).parent / "scheme_matching_evaluation_dataset.json")

    if not eval_path.exists():
        print(f"❌ Error: Evaluation dataset not found at '{eval_path}'")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("\n==================================================")
    print("📜 GOVERNMENT SCHEME MATCHING ENGINE EVALUATION")
    print("==================================================")
    print(f"Evaluation Dataset: {eval_path.name} ({len(eval_items)} artisan profiles)")
    print("--------------------------------------------------")

    engine = SchemeMatchingEngine()

    top_1_hits = 0
    top_3_hits = 0
    precision_k_hits = 0
    source_verified_count = 0
    total_schemes_recommended = 0
    total_queries = len(eval_items)

    results = []

    for item in eval_items:
        eval_id = item.get("eval_id")
        desc = item.get("description")
        req_dict = item.get("request", {})
        request = SchemeMatchingRequest.model_validate(req_dict)

        expected_top_id = item.get("expected_top_scheme_id")
        expected_top_3_ids = item.get("expected_top_3_scheme_ids", [expected_top_id])
        min_expected_count = item.get("expected_matched_count_min", 1)

        try:
            res = engine.match_schemes(request)
            top_results = res.results

            actual_top_id = top_results[0].scheme.scheme_id if top_results else None
            actual_top_name = top_results[0].scheme.scheme_name if top_results else "None"
            actual_top_3_ids = [r.scheme.scheme_id for r in top_results[:3]]

            # Top-1 Match check
            is_top_1 = (actual_top_id in [expected_top_id] + expected_top_3_ids[:1])
            if is_top_1:
                top_1_hits += 1

            # Top-3 Match check
            is_top_3 = (expected_top_id in actual_top_3_ids) or any(sid in expected_top_3_ids for sid in actual_top_3_ids)
            if is_top_3:
                top_3_hits += 1

            # Precision @ K check
            is_precision_hit = (len(top_results) >= min_expected_count)
            if is_precision_hit:
                precision_k_hits += 1

            # Source Verification check
            for r in top_results:
                total_schemes_recommended += 1
                if r.scheme.official_source_url and r.scheme.official_source_url.startswith("http"):
                    source_verified_count += 1

            top_score = top_results[0].match_score if top_results else 0.0

            print(f"[{eval_id}] {desc}")
            print(f"   Profile: Craft='{request.craft_type}', Location='{request.location}', Business='{request.business_status}'")
            print(f"   Expected Top: {expected_top_id} | Actual Top: {actual_top_id} ({actual_top_name})")
            print(f"   Match Score: {top_score} | Total Matched: {res.total_matched}")
            print("--------------------------------------------------")

            results.append({
                "eval_id": eval_id,
                "description": desc,
                "expected_top_scheme_id": expected_top_id,
                "actual_top_scheme_id": actual_top_id,
                "top_1_hit": is_top_1,
                "top_3_hit": is_top_3,
                "precision_k_hit": is_precision_hit,
                "total_matched": res.total_matched,
            })

        except CatalogAIException as e:
            print(f"❌ Scheme Matching Error on item [{eval_id}]: {e.message}")
        except Exception as e:
            print(f"❌ Unexpected Error on item [{eval_id}]: {str(e)}")

    top_1_accuracy = (top_1_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    top_3_accuracy = (top_3_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    precision_k_accuracy = (precision_k_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    source_verification_rate = (source_verified_count / total_schemes_recommended * 100.0) if total_schemes_recommended > 0 else 0.0

    print("\n📊 EVALUATION SUMMARY (OFFICIAL GOVERNMENT SCHEMES DATASET):")
    print("==================================================")
    print(f"🎯 Top-1 Scheme Match Accuracy:      {top_1_accuracy:.1f}% ({top_1_hits}/{total_queries})")
    print(f"🎯 Top-3 Scheme Match Accuracy:      {top_3_accuracy:.1f}% ({top_3_hits}/{total_queries})")
    print(f"🎯 Precision @ K Scheme Match:        {precision_k_accuracy:.1f}% ({precision_k_hits}/{total_queries})")
    print(f"🔗 Official Source Verification Rate: {source_verification_rate:.1f}% ({source_verified_count}/{total_schemes_recommended})")
    print("==================================================")

    output_file = Path(__file__).parent / "scheme_matching_evaluation_summary.json"
    summary_data = {
        "top_1_scheme_match_accuracy_percent": round(top_1_accuracy, 2),
        "top_3_scheme_match_accuracy_percent": round(top_3_accuracy, 2),
        "precision_at_k_percent": round(precision_k_accuracy, 2),
        "official_source_verification_rate_percent": round(source_verification_rate, 2),
        "total_test_profiles": total_queries,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Full summary saved to: {output_file.name}")
    return summary_data



if __name__ == "__main__":
    evaluate_scheme_matching_dataset()
