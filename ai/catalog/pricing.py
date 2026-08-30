"""
Smart Pricing Intelligence AI Engine for Artisan Marketplace.
Combines Financial Cost-Plus Artisan Margin Protection with Multimodal Market Comparable Matching & Gemini 3.6 Flash.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from config import Config
from exceptions import (
    PricingError,
    MissingPricingInputError,
    InvalidPricingInputError,
    PricingDatasetError,
    MissingAPIKeyError,
    AIServiceError,
    InvalidAIResponseError,
)
from schema import (
    ProductCatalog,
    PricingInput,
    PriceRange,
    CostBreakdown,
    PricingEstimateResponse,
)

logger = logging.getLogger("pricing_ai_engine")

PRICING_PROMPT_TEMPLATE = """You are an expert AI Pricing Intelligence Engine specializing in Indian handcrafted artisanal products.
Analyze the following product catalog metadata, financial cost floor, and historical marketplace benchmark data to determine an optimal, fair, and competitive selling price.

Product Catalog Metadata:
- Product Name: {product_name}
- Category: {category}
- Craft Type: {craft_type}
- Materials: {materials}
- Dominant Colors: {colors}
- Search Tags: {tags}
- Catalog Extraction Confidence: {catalog_confidence}

Financial Cost Floor Breakdown (Artisan Protection):
- Raw Material Cost: ₹ {material_cost}
- Labor Hours Spent: {labor_hours} hours
- Artisan Hourly Wage Rate: ₹ {hourly_wage}/hr
- Labor Subtotal: ₹ {labor_cost}
- Packaging & Overhead: ₹ {overhead}
- Minimum Viable Cost Floor (Cost + 15% Min Artisan Margin): ₹ {cost_floor}

Target Market Tier: {market_tier}

Historical Marketplace Comparable Benchmarks ({comps_count} items matched):
{comps_summary}

Instructions:
1. Recommend a fair selling price (recommended_price_inr) in Indian Rupees (₹ INR).
2. The recommended price MUST NOT be lower than the minimum viable cost floor (₹ {cost_floor}) to guarantee fair artisan profit.
3. Provide a realistic price range (min_price_inr, recommended_price_inr, max_price_inr).
4. Provide a clear, buyer-friendly and artisan-reassuring pricing explanation rationale.
5. Return ONLY valid JSON matching this exact structure:
{{
  "recommended_price_inr": <float>,
  "min_price_inr": <float>,
  "max_price_inr": <float>,
  "pricing_explanation": "<clear explanation string>"
}}
"""


class PricingEngine:
    """Core AI engine for artisan product pricing intelligence."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dataset_path: Optional[Path] = None
    ):
        self._api_key = api_key
        self._model_name = model_name or Config.get_model_name()
        self._dataset_path = dataset_path or (Path(__file__).parent / "marketplace_dataset.json")
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return Config.get_api_key(strict=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads historical marketplace benchmark dataset from JSON."""
        if self._dataset is not None:
            return self._dataset

        if not self._dataset_path.exists():
            logger.error(f"Marketplace dataset file not found at: {self._dataset_path}")
            raise PricingDatasetError(f"Marketplace benchmark dataset file not found at {self._dataset_path}")

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

    def validate_input(self, input_data: PricingInput):
        """Validates input parameters for non-negativity and presence."""
        if not input_data or not input_data.catalog:
            raise MissingPricingInputError("Pricing input must contain a valid ProductCatalog object.")
        if input_data.material_cost_inr < 0:
            raise InvalidPricingInputError("Material cost cannot be negative.")
        if input_data.labor_hours < 0:
            raise InvalidPricingInputError("Labor hours cannot be negative.")
        if input_data.artisan_hourly_wage_inr < 0:
            raise InvalidPricingInputError("Hourly wage rate cannot be negative.")
        if input_data.packaging_shipping_cost_inr < 0:
            raise InvalidPricingInputError("Packaging cost cannot be negative.")

    def calculate_cost_breakdown(self, input_data: PricingInput) -> CostBreakdown:
        """
        Calculates itemized cost breakdown and financial profit floor.
        Cost Floor = Material Cost + Labor Cost + Overhead + 15% Minimum Artisan Profit Margin.
        """
        mat = round(float(input_data.material_cost_inr), 2)
        hrs = round(float(input_data.labor_hours), 2)
        wage = round(float(input_data.artisan_hourly_wage_inr), 2)
        overhead = round(float(input_data.packaging_shipping_cost_inr), 2)

        labor_cost = round(hrs * wage, 2)
        base_cost = mat + labor_cost + overhead
        artisan_margin = round(base_cost * 0.15, 2)
        cost_floor = round(base_cost + artisan_margin, 2)

        return CostBreakdown(
            material_cost_inr=mat,
            labor_cost_inr=labor_cost,
            overhead_inr=overhead,
            artisan_margin_inr=artisan_margin,
            cost_floor_inr=cost_floor,
        )

    def compute_similarity(self, catalog: ProductCatalog, comp: Dict[str, Any]) -> float:
        """Computes similarity score between input product catalog and historical marketplace benchmark item."""
        score = 0.0

        # 1. Craft match (Weight 0.40)
        input_craft = (catalog.craft_type or "").strip().lower()
        comp_craft = str(comp.get("craft_type", "")).strip().lower()
        if input_craft and comp_craft and (input_craft in comp_craft or comp_craft in input_craft):
            score += 0.40

        # 2. Category match (Weight 0.30)
        input_cat = (catalog.category or "").strip().lower()
        comp_cat = str(comp.get("category", "")).strip().lower()
        if input_cat and comp_cat and (input_cat in comp_cat or comp_cat in input_cat):
            score += 0.30

        # 3. Material overlap Jaccard score (Weight 0.15)
        input_mats = set(m.lower().strip() for m in catalog.materials if m)
        comp_mats = set(m.lower().strip() for m in comp.get("materials", []) if m)
        if input_mats and comp_mats:
            jaccard_mats = len(input_mats & comp_mats) / len(input_mats | comp_mats)
            score += 0.15 * jaccard_mats

        # 4. Tag overlap Jaccard score (Weight 0.15)
        input_tags = set(t.lower().strip() for t in catalog.tags if t)
        comp_tags = set(t.lower().strip() for t in comp.get("tags", []) if t)
        if input_tags and comp_tags:
            jaccard_tags = len(input_tags & comp_tags) / len(input_tags | comp_tags)
            score += 0.15 * jaccard_tags

        return round(score, 3)

    def find_comparable_products(self, catalog: ProductCatalog, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves top K matched marketplace comps ordered by similarity score."""
        dataset = self.load_dataset()
        scored_items = []
        for item in dataset:
            sim = self.compute_similarity(catalog, item)
            item_with_score = dict(item)
            item_with_score["similarity_score"] = sim
            scored_items.append(item_with_score)

        scored_items.sort(key=lambda x: x["similarity_score"], reverse=True)
        # Filter items with non-zero similarity score if available
        matched = [it for it in scored_items if it["similarity_score"] > 0]
        return matched[:top_k] if matched else scored_items[:top_k]

    def calculate_confidence_score(
        self,
        catalog_confidence: float,
        comps_matched: List[Dict[str, Any]]
    ) -> float:
        """
        Calculates pricing confidence score based on catalog confidence, density of matched comps,
        and price variance.
        """
        density_score = min(len(comps_matched) / 5.0, 1.0)
        
        # Calculate price variance coefficient among comps
        if comps_matched:
            medians = [float(c.get("median_price_inr", 0.0)) for c in comps_matched if c.get("median_price_inr")]
            if medians and len(medians) > 1:
                mean_p = sum(medians) / len(medians)
                std_p = (sum((x - mean_p) ** 2 for x in medians) / len(medians)) ** 0.5
                cv = std_p / mean_p if mean_p > 0 else 0.0
                spread_score = max(0.0, 1.0 - cv)
            else:
                spread_score = 0.8
        else:
            spread_score = 0.5

        final_conf = (0.40 * density_score) + (0.30 * spread_score) + (0.30 * catalog_confidence)
        return round(max(0.10, min(1.0, final_conf)), 2)

    def estimate_price(self, input_data: PricingInput) -> PricingEstimateResponse:
        """
        Generates structured price recommendation using financial cost floor calculation,
        marketplace benchmark comps retrieval, and Gemini 3.6 Flash synthesis.
        """
        self.validate_input(input_data)
        cost_breakdown = self.calculate_cost_breakdown(input_data)
        comps = self.find_comparable_products(input_data.catalog, top_k=5)
        api_key = self._get_api_key()

        # Build comparable items summary text
        comps_text_lines = []
        comp_medians = []
        for i, c in enumerate(comps, 1):
            med = c.get("median_price_inr", 0)
            comp_medians.append(med)
            comps_text_lines.append(
                f"{i}. {c.get('product_name')} ({c.get('craft_type')}, {c.get('category')}): "
                f"₹{c.get('min_price_inr')}-₹{c.get('max_price_inr')} (Median: ₹{med}) [{c.get('platform')}]"
            )
        comps_summary = "\n".join(comps_text_lines) if comps_text_lines else "No direct comps found."

        # Format Gemini Prompt
        prompt = PRICING_PROMPT_TEMPLATE.format(
            product_name=input_data.catalog.product_name,
            category=input_data.catalog.category,
            craft_type=input_data.catalog.craft_type,
            materials=", ".join(input_data.catalog.materials),
            colors=", ".join(input_data.catalog.colors),
            tags=", ".join(input_data.catalog.tags),
            catalog_confidence=input_data.catalog.confidence_score,
            material_cost=cost_breakdown.material_cost_inr,
            labor_hours=input_data.labor_hours,
            hourly_wage=input_data.artisan_hourly_wage_inr,
            labor_cost=cost_breakdown.labor_cost_inr,
            overhead=cost_breakdown.overhead_inr,
            cost_floor=cost_breakdown.cost_floor_inr,
            market_tier=input_data.market_tier or "Fair Trade / Artisanal",
            comps_count=len(comps),
            comps_summary=comps_summary,
        )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )

            logger.info(f"Executing pricing engine call using model: '{self._model_name}'")
            response = client.models.generate_content(
                model=self._model_name,
                contents=[prompt],
                config=config,
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise InvalidAIResponseError("Gemini Pricing API returned an empty response.")

        except (MissingAPIKeyError, MissingPricingInputError, InvalidPricingInputError):
            raise
        except Exception as e:
            if "GEMINI_API_KEY" in str(e):
                raise MissingAPIKeyError(str(e))
            logger.error(f"Error during Gemini Pricing API call with model '{self._model_name}': {str(e)}", exc_info=True)
            raise AIServiceError(f"Pricing engine model '{self._model_name}' call failed: {str(e)}")

        # Parse output JSON into response object
        try:
            cleaned_text = raw_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()

            parsed = json.loads(cleaned_text)
            rec_price = float(parsed.get("recommended_price_inr", 0.0))
            min_price = float(parsed.get("min_price_inr", 0.0))
            max_price = float(parsed.get("max_price_inr", 0.0))
            explanation = str(parsed.get("pricing_explanation", "")).strip()

            # Enforce Financial Floor Protection rule: recommended_price >= cost_floor
            floor = cost_breakdown.cost_floor_inr
            if rec_price < floor:
                rec_price = floor
            if min_price < floor:
                min_price = floor
            if max_price < rec_price:
                max_price = round(rec_price * 1.25, 2)

            conf_score = self.calculate_confidence_score(
                catalog_confidence=input_data.catalog.confidence_score,
                comps_matched=comps
            )

            return PricingEstimateResponse(
                recommended_price_inr=round(rec_price, 2),
                price_range=PriceRange(
                    min_price_inr=round(min_price, 2),
                    recommended_price_inr=round(rec_price, 2),
                    max_price_inr=round(max_price, 2),
                ),
                cost_breakdown=cost_breakdown,
                comparable_products_count=len(comps),
                confidence_score=conf_score,
                pricing_explanation=explanation or (
                    f"Recommended price of ₹{rec_price:.2f} covers the artisan cost floor of ₹{floor:.2f} "
                    f"and provides a fair profit margin."
                )
            )

        except Exception as e:
            logger.error(f"Failed to parse pricing engine response: {raw_text}", exc_info=True)
            raise InvalidAIResponseError(f"Failed to parse pricing engine output JSON: {str(e)}")
