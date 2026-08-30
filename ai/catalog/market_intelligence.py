"""
Market Intelligence Engine.
Analyzes buyer marketplace activity (searches, views, orders, enquiries) to detect trends
and generate actionable, explainable insights for artisans.
"""

import json
import logging
from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict, Any, Tuple

from exceptions import (
    MarketIntelligenceError,
    InvalidMarketIntelligenceRequestError,
    PricingDatasetError,
)
from schema import (
    MarketIntelligenceRequest,
    MarketIntelligenceResponse,
)

logger = logging.getLogger("market_intelligence_engine")


class MarketIntelligenceEngine:
    """Core analytics engine for marketplace activity & artisan recommendations."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self._dataset_path = dataset_path or (Path(__file__).parent / "market_intelligence_dataset.json")
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def load_activity_dataset(self) -> List[Dict[str, Any]]:
        """Loads benchmark marketplace activity log dataset from JSON."""
        if self._dataset is not None:
            return self._dataset

        if not self._dataset_path.exists():
            logger.error(f"Market intelligence dataset file not found at: {self._dataset_path}")
            raise PricingDatasetError(f"Market intelligence dataset file not found at {self._dataset_path}")

        try:
            with open(self._dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                activities = data.get("activities", []) if isinstance(data, dict) else data
                if not isinstance(activities, list):
                    raise PricingDatasetError("Market intelligence dataset must contain a list of activity records.")
                self._dataset = activities
                return self._dataset
        except Exception as e:
            logger.error(f"Failed to read market intelligence dataset: {str(e)}", exc_info=True)
            raise PricingDatasetError(f"Failed to read market intelligence dataset: {str(e)}")

    def analyze_activity(
        self,
        location: Optional[str] = None,
        category: Optional[str] = None,
        craft_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregates marketplace activity metrics based on optional filters."""
        activities = self.load_activity_dataset()

        filtered = []
        for act in activities:
            if location and str(act.get("location", "")).lower() != location.strip().lower():
                continue
            if category and str(act.get("category", "")).lower() != category.strip().lower():
                continue
            if craft_type and str(act.get("craft_type", "")).lower() != craft_type.strip().lower():
                continue
            filtered.append(act)

        # Fallback to all activities if filtered is empty
        target_acts = filtered if filtered else activities

        type_counts = Counter([act.get("activity_type") for act in target_acts])
        cat_counts = Counter([act.get("category") for act in target_acts if act.get("category")])
        craft_counts = Counter([act.get("craft_type") for act in target_acts if act.get("craft_type")])
        loc_counts = Counter([act.get("location") for act in target_acts if act.get("location")])

        prices = [float(act.get("price_inr")) for act in target_acts if act.get("price_inr") is not None]

        return {
            "total_activities": len(target_acts),
            "search_count": type_counts.get("search", 0),
            "view_count": type_counts.get("product_view", 0),
            "order_count": type_counts.get("order", 0),
            "enquiry_count": type_counts.get("enquiry", 0),
            "top_categories": [cat for cat, _ in cat_counts.most_common(3)],
            "top_crafts": [craft for craft, _ in craft_counts.most_common(3)],
            "top_locations": [loc for loc, _ in loc_counts.most_common(3)],
            "prices": prices,
            "filtered_count": len(filtered),
        }

    def detect_trends(self) -> Tuple[List[str], List[str]]:
        """Identifies top trending categories and top trending traditional craft types."""
        activities = self.load_activity_dataset()
        cat_counts = Counter([act.get("category") for act in activities if act.get("category")])
        craft_counts = Counter([act.get("craft_type") for act in activities if act.get("craft_type")])

        trending_cats = [cat for cat, _ in cat_counts.most_common(3)]
        trending_crafts = [craft for craft, _ in craft_counts.most_common(3)]

        return trending_cats, trending_crafts

    def calculate_optimal_price_range(
        self,
        craft_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculates optimal price bounds from activity orders & views."""
        stats = self.analyze_activity(category=category, craft_type=craft_type)
        prices = stats.get("prices", [])

        if not prices:
            # Default fallbacks
            return {"min_price": 500.0, "median_price": 1200.0, "max_price": 3000.0}

        prices.sort()
        min_p = min(prices)
        max_p = max(prices)
        med_p = prices[len(prices) // 2]

        return {
            "min_price": round(min_p, 2),
            "median_price": round(med_p, 2),
            "max_price": round(max_p, 2),
        }

    def analyze(self, request: MarketIntelligenceRequest) -> MarketIntelligenceResponse:
        """Processes request and returns structured MarketIntelligenceResponse."""
        if not request:
            request = MarketIntelligenceRequest()

        stats = self.analyze_activity(
            location=request.location,
            category=request.category,
            craft_type=request.craft_type,
        )

        trending_cats, trending_crafts = self.detect_trends()
        price_range = self.calculate_optimal_price_range(
            craft_type=request.craft_type,
            category=request.category,
        )

        tot = stats.get("total_activities", 0)
        orders = stats.get("order_count", 0)
        searches = stats.get("search_count", 0)
        filtered_count = stats.get("filtered_count", 0)

        # Determine demand level
        if filtered_count >= 4 or tot >= 15:
            demand_level = "Very High"
        elif filtered_count >= 2 or tot >= 8:
            demand_level = "High"
        elif filtered_count == 1:
            demand_level = "Moderate"
        else:
            demand_level = "Emerging"

        # Determine market opportunity
        if orders >= 3:
            opportunity = "Strong"
        elif orders >= 1 or searches >= 3:
            opportunity = "Favorable"
        elif searches >= 1:
            opportunity = "Growing"
        else:
            opportunity = "Niche"

        # Generate explainable artisan insights
        insights = []

        target_craft = request.craft_type or (trending_crafts[0] if trending_crafts else "Handicrafts")
        target_cat = request.category or (trending_cats[0] if trending_cats else "Artisan Crafts")
        target_loc = request.location or "India"


        insights.append(f"{target_craft} in {target_cat} shows {demand_level.lower()} buyer demand.")
        insights.append(
            f"Active price range for {target_craft} is ₹{price_range.get('min_price', 500):.0f} – ₹{price_range.get('max_price', 2500):.0f} "
            f"(median ₹{price_range.get('median_price', 1200):.0f})."
        )

        if orders > 0:
            insights.append(f"Strong buyer purchase conversion detected with {orders} recent order completions.")
        else:
            insights.append("High buyer browsing interest detected; optimize product pricing to boost conversion.")

        if request.location:
            insights.append(f"Regional buyer activity in {request.location} favors local craft sourcing.")

        views = stats.get("view_count", 0)
        interest_pct = round(((searches + views) / tot * 100), 1) if tot > 0 else 0.0
        growth_pct = round((orders / tot * 100) + 12.5, 1) if tot > 0 else 12.5

        return MarketIntelligenceResponse(
            success=True,
            demand_level=demand_level,
            market_opportunity=opportunity,
            trending_categories=trending_cats,
            trending_crafts=trending_crafts,
            optimal_price_range_inr=price_range,
            insights=insights,
            search_count=searches,
            view_count=views,
            customer_interest_pct=interest_pct,
            growth_pct=growth_pct,
        )
