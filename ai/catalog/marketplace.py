"""
Buyer Marketplace & Artisan Matching Engine.
Provides deterministic, explainable search and recommendation ranking for artisan products.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from exceptions import (
    MarketplaceError,
    InvalidMarketplaceQueryError,
    PricingDatasetError,
)
from schema import (
    MarketplaceSearchQuery,
    ArtisanInfo,
    MarketplaceProductItem,
    MarketplaceMatchResult,
    MarketplaceSearchResponse,
)

logger = logging.getLogger("marketplace_engine")


class MarketplaceEngine:
    """Core search and recommendation engine for buyer marketplace."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self._dataset_path = dataset_path or (Path(__file__).parent / "marketplace_dataset.json")
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads marketplace dataset from JSON."""
        if self._dataset is not None:
            return self._dataset

        if not self._dataset_path.exists():
            logger.error(f"Marketplace dataset file not found at: {self._dataset_path}")
            raise PricingDatasetError(f"Marketplace dataset file not found at {self._dataset_path}")

        try:
            with open(self._dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise PricingDatasetError("Marketplace dataset must contain a list of product records.")
                self._dataset = data
                return self._dataset
        except Exception as e:
            logger.error(f"Failed to read marketplace dataset: {str(e)}", exc_info=True)
            raise PricingDatasetError(f"Failed to read marketplace dataset: {str(e)}")

    def validate_query(self, search_query: MarketplaceSearchQuery):
        """Validates search query parameters."""
        if not search_query:
            return

        if search_query.price_min is not None and search_query.price_min < 0:
            raise InvalidMarketplaceQueryError("price_min cannot be negative.")

        if search_query.price_max is not None and search_query.price_max < 0:
            raise InvalidMarketplaceQueryError("price_max cannot be negative.")

        if (
            search_query.price_min is not None
            and search_query.price_max is not None
            and search_query.price_min > search_query.price_max
        ):
            raise InvalidMarketplaceQueryError("price_min cannot be greater than price_max.")

    def score_product(
        self,
        product_dict: Dict[str, Any],
        search_query: MarketplaceSearchQuery
    ) -> Tuple[float, List[str]]:
        """
        Calculates relevance score (0.0 to 1.0) and generates human-readable match reasons.
        Uses active weight normalization across specified query parameters:
        - Category match: 25% (0.25)
        - Keyword match: 25% (0.25)
        - Craft type match: 15% (0.15)
        - Location match: 10% (0.10)
        - Product similarity: 15% (0.15)
        - Price compatibility: 10% (0.10)
        """
        reasons = []

        q_cat = (search_query.category or "").strip().lower()
        q_kw = (search_query.query or "").strip().lower()
        q_craft = (search_query.craft_type or "").strip().lower()
        q_loc = (search_query.location or "").strip().lower()
        p_min = search_query.price_min
        p_max = search_query.price_max

        p_name = str(product_dict.get("product_name", "")).strip()
        p_cat = str(product_dict.get("category", "")).strip()
        p_craft = str(product_dict.get("craft_type", "")).strip()
        p_desc = str(product_dict.get("description", "")).strip()
        p_price = float(product_dict.get("price_inr", product_dict.get("median_price_inr", 0.0)))
        p_mats = [str(m).lower() for m in product_dict.get("materials", [])]
        p_tags = [str(t).lower() for t in product_dict.get("tags", [])]

        artisan_data = product_dict.get("artisan", {})
        p_art_loc = str(artisan_data.get("location", "")).strip()

        has_any_filter = any([q_cat, q_kw, q_craft, q_loc, p_min is not None, p_max is not None])

        if not has_any_filter:
            return 1.0, ["Artisan product listing"]

        active_weight_sum = 0.0
        weighted_score_sum = 0.0

        # 1. Category Match (25%)
        if q_cat:
            active_weight_sum += 0.25
            if q_cat == p_cat.lower() or q_cat in p_cat.lower() or p_cat.lower() in q_cat:
                s_cat = 1.0
                reasons.append(f"Same craft category ('{p_cat}')")
            else:
                s_cat = 0.0
                return 0.0, []  # Hard filter out non-matching category
            weighted_score_sum += 0.25 * s_cat

        # 2. Keyword Match (25%)
        if q_kw:
            active_weight_sum += 0.25
            kw_tokens = [t for t in q_kw.split() if len(t) > 1]
            if not kw_tokens:
                s_kw = 1.0
            else:
                target_text = f"{p_name} {p_desc} {p_craft} {p_cat} {' '.join(p_mats)} {' '.join(p_tags)}".lower()
                matches = sum(1 for tok in kw_tokens if tok in target_text)
                s_kw = matches / len(kw_tokens)
                if matches > 0:
                    reasons.append("Keyword matched")
            weighted_score_sum += 0.25 * s_kw

        # 3. Craft Type Match (15%)
        if q_craft:
            active_weight_sum += 0.15
            if q_craft == p_craft.lower() or q_craft in p_craft.lower() or p_craft.lower() in q_craft:
                s_craft = 1.0
                reasons.append(f"Craft type matched ('{p_craft}')")
            else:
                s_craft = 0.0
                return 0.0, []  # Hard filter out non-matching craft type
            weighted_score_sum += 0.15 * s_craft

        # 4. Location Match (10%)
        if q_loc:
            active_weight_sum += 0.10
            if q_loc in p_art_loc.lower() or p_art_loc.lower() in q_loc:
                s_loc = 1.0
                reasons.append("Available in selected location")
            else:
                s_loc = 0.0
                return 0.0, []  # Hard filter out non-matching location
            weighted_score_sum += 0.10 * s_loc

        # 5. Product Similarity (15%)
        if q_kw or q_cat or q_craft:
            active_weight_sum += 0.15
            query_words = set(q_kw.split() + q_cat.split() + q_craft.split())
            query_words = {w for w in query_words if len(w) > 1}
            prod_words = set(p_tags + p_mats + [w.lower() for w in p_name.split() if len(w) > 1])
            if query_words and prod_words:
                jaccard = len(query_words & prod_words) / len(query_words | prod_words)
                s_sim = min(1.0, jaccard * 2.0)
            else:
                s_sim = 0.5
            if s_sim >= 0.2:
                reasons.append("Similar product")
            weighted_score_sum += 0.15 * s_sim

        # 6. Price Compatibility (10%)
        if p_min is not None or p_max is not None:
            active_weight_sum += 0.10
            lower = p_min if p_min is not None else 0.0
            upper = p_max if p_max is not None else float("inf")
            if lower <= p_price <= upper:
                s_price = 1.0
                reasons.append("Within preferred price range")
            else:
                return 0.0, []  # Hard filter out products outside price range
            weighted_score_sum += 0.10 * s_price


        if active_weight_sum <= 0:
            return 1.0, ["Artisan product listing"]

        final_score = weighted_score_sum / active_weight_sum
        unique_reasons = list(dict.fromkeys(reasons))
        if not unique_reasons:
            unique_reasons = ["Artisan product listing"]

        return round(final_score, 2), unique_reasons


    def search_products(self, search_query: MarketplaceSearchQuery) -> MarketplaceSearchResponse:
        """Searches products, ranks by relevance score, and returns MarketplaceSearchResponse."""
        self.validate_query(search_query)
        dataset = self.load_dataset()

        match_results: List[MarketplaceMatchResult] = []

        for prod_dict in dataset:
            score, reasons = self.score_product(prod_dict, search_query)

            # Filter out products with 0 relevance if specific query/filters were provided
            has_filters = any([
                search_query.query,
                search_query.category,
                search_query.craft_type,
                search_query.location,
                search_query.price_min is not None,
                search_query.price_max is not None,
            ])

            if has_filters and score < 0.15:
                continue

            artisan_dict = prod_dict.get("artisan", {})
            artisan = ArtisanInfo(
                artisan_id=str(artisan_dict.get("artisan_id", "ART001")),
                artisan_name=str(artisan_dict.get("artisan_name", "Artisan Collective")),
                location=str(artisan_dict.get("location", "India")),
            )

            product_item = MarketplaceProductItem(
                product_id=str(prod_dict.get("product_id", "")),
                product_name=str(prod_dict.get("product_name", "")),
                category=str(prod_dict.get("category", "")),
                craft_type=str(prod_dict.get("craft_type", "")),
                materials=list(prod_dict.get("materials", [])),
                description=str(prod_dict.get("description", "")),
                price_inr=float(prod_dict.get("price_inr", prod_dict.get("median_price_inr", 0.0))),
                artisan=artisan,
                tags=list(prod_dict.get("tags", [])),
            )

            match_results.append(
                MarketplaceMatchResult(
                    product=product_item,
                    relevance_score=score,
                    match_reasons=reasons,
                )
            )

        # Sort by relevance_score descending
        match_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return MarketplaceSearchResponse(
            success=True,
            total_results=len(match_results),
            results=match_results,
        )
