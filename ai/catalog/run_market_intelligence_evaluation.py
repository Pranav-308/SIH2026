"""
CLI evaluation script for Market Intelligence Engine.

Usage:
  python run_market_intelligence_evaluation.py
"""

import sys
import json
import logging
from pathlib import Path

# Ensure ai/catalog is in import path
sys.path.insert(0, str(Path(__file__).parent))

from schema import MarketIntelligenceRequest
from market_intelligence import MarketIntelligenceEngine
from exceptions import CatalogAIException

logger = logging.getLogger("market_intelligence_evaluation")


def evaluate_market_intelligence_dataset(eval_dataset_path: Path = None):
    eval_path = eval_dataset_path or (Path(__file__).parent / "market_intelligence_evaluation_dataset.json")

    if not eval_path.exists():
        print(f"❌ Error: Evaluation dataset not found at '{eval_path}'")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    print("\n==================================================")
    print("📊 MARKET INTELLIGENCE ENGINE EVALUATION")
    print("==================================================")
    print(f"Evaluation Dataset: {eval_path.name} ({len(eval_items)} test queries)")
    print("--------------------------------------------------")

    engine = MarketIntelligenceEngine()

    trend_id_hits = 0
    cat_demand_hits = 0
    insight_relevance_hits = 0
    total_queries = len(eval_items)

    results = []

    for item in eval_items:
        eval_id = item.get("eval_id")
        desc = item.get("description")
        req_dict = item.get("request", {})
        request = MarketIntelligenceRequest.model_validate(req_dict)

        expected_demand = item.get("expected_demand_level")
        expected_opportunity = item.get("expected_market_opportunity")
        expected_cat = item.get("expected_trending_category")

        try:
            res = engine.analyze(request)

            # Trend Identification check
            is_trend_hit = (expected_cat in res.trending_categories)
            if is_trend_hit:
                trend_id_hits += 1

            # Category Demand check
            is_demand_hit = (res.demand_level in ["Very High", "High", "Moderate"])
            if is_demand_hit:
                cat_demand_hits += 1

            # Insight Relevance check
            is_insight_hit = (len(res.insights) >= 3 and any(request.craft_type in ins for ins in res.insights if request.craft_type))
            if is_insight_hit:
                insight_relevance_hits += 1

            print(f"[{eval_id}] {desc}")
            print(f"   Request: {req_dict}")
            print(f"   Demand Level: {res.demand_level} | Opportunity: {res.market_opportunity}")
            print(f"   Trending Categories: {res.trending_categories}")
            print(f"   Insights Count: {len(res.insights)}")
            print("--------------------------------------------------")

            results.append({
                "eval_id": eval_id,
                "description": desc,
                "demand_level": res.demand_level,
                "market_opportunity": res.market_opportunity,
                "trend_identification_hit": is_trend_hit,
                "category_demand_hit": is_demand_hit,
                "insight_relevance_hit": is_insight_hit,
            })

        except CatalogAIException as e:
            print(f"❌ Market Intelligence Error on item [{eval_id}]: {e.message}")
        except Exception as e:
            print(f"❌ Unexpected Error on item [{eval_id}]: {str(e)}")

    trend_accuracy = (trend_id_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    demand_accuracy = (cat_demand_hits / total_queries * 100.0) if total_queries > 0 else 0.0
    insight_accuracy = (insight_relevance_hits / total_queries * 100.0) if total_queries > 0 else 0.0

    print("\n📊 EVALUATION SUMMARY (BENCHMARK ACTIVITY DATASET):")
    print("==================================================")
    print(f"📈 Trend Identification Accuracy: {trend_accuracy:.1f}% ({trend_id_hits}/{total_queries})")
    print(f"🎯 Category Demand Accuracy:       {demand_accuracy:.1f}% ({cat_demand_hits}/{total_queries})")
    print(f"💡 Artisan Insight Relevance Rate:  {insight_accuracy:.1f}% ({insight_relevance_hits}/{total_queries})")
    print("==================================================")

    output_file = Path(__file__).parent / "market_intelligence_evaluation_summary.json"
    summary_data = {
        "trend_identification_accuracy_percent": round(trend_accuracy, 2),
        "category_demand_accuracy_percent": round(demand_accuracy, 2),
        "artisan_insight_relevance_rate_percent": round(insight_accuracy, 2),
        "total_test_queries": total_queries,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Full summary saved to: {output_file.name}")
    return summary_data


if __name__ == "__main__":
    evaluate_market_intelligence_dataset()
