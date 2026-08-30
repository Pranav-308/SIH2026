"""
Government Scheme Matching Engine.
Provides deterministic, explainable rule-based matching connecting artisans with verified official Government schemes.
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from exceptions import (
    SchemeMatchingError,
    InvalidSchemeRequestError,
    PricingDatasetError,
)
from schema import (
    GovernmentSchemeItem,
    SchemeMatchingRequest,
    MatchedSchemeResult,
    SchemeMatchingResponse,
)

logger = logging.getLogger("scheme_matching_engine")


class SchemeMatchingEngine:
    """Core rule-based matching engine for Government Schemes."""

    def __init__(self, dataset_path: Optional[Path] = None):
        self._dataset_path = dataset_path or (Path(__file__).parent / "schemes_dataset.json")
        self._dataset: Optional[List[Dict[str, Any]]] = None

    def load_schemes_dataset(self) -> List[Dict[str, Any]]:
        """Loads verified official government schemes dataset from JSON."""
        if self._dataset is not None:
            return self._dataset

        if not self._dataset_path.exists():
            logger.error(f"Schemes dataset file not found at: {self._dataset_path}")
            raise PricingDatasetError(f"Schemes dataset file not found at {self._dataset_path}")

        try:
            with open(self._dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise PricingDatasetError("Schemes dataset must contain a list of scheme records.")
                self._dataset = data
                return self._dataset
        except Exception as e:
            logger.error(f"Failed to read schemes dataset: {str(e)}", exc_info=True)
            raise PricingDatasetError(f"Failed to read schemes dataset: {str(e)}")

    def validate_request(self, request: SchemeMatchingRequest):
        """Validates incoming scheme matching request parameters."""
        if not request:
            raise InvalidSchemeRequestError("Scheme matching request cannot be empty.")
        if not request.craft_type or not request.craft_type.strip():
            raise InvalidSchemeRequestError("craft_type parameter is required.")
        if not request.location or not request.location.strip():
            raise InvalidSchemeRequestError("location parameter is required.")

    def match_scheme_for_profile(
        self,
        scheme_dict: Dict[str, Any],
        request: SchemeMatchingRequest
    ) -> Tuple[float, List[str], List[str]]:
        """
        Evaluates an artisan profile against a single government scheme.
        Returns (match_score, matched_criteria, unmet_criteria).
        """
        matched = []
        unmet = []

        req_craft = request.craft_type.strip().lower()
        req_loc = request.location.strip().lower()
        req_bus = (request.business_status or "Individual Artisan").strip().lower()
        req_reg = (request.registration_status or "Unregistered / Informal").strip().lower()

        t_crafts = [str(c).lower() for c in scheme_dict.get("target_crafts", [])]
        t_locs = [str(l).lower() for l in scheme_dict.get("target_locations", [])]
        e_buses = [str(b).lower() for b in scheme_dict.get("eligible_business_statuses", [])]
        e_regs = [str(r).lower() for r in scheme_dict.get("eligible_registration_statuses", [])]

        # 1. Location Match (25%)
        if any(req_loc in l or l in req_loc for l in t_locs if l != "all india"):
            s_loc = 1.0
            matched.append(f"State location '{request.location}' matches specific state scheme coverage")
        elif "all india" in t_locs or "all india" in [l.lower() for l in t_locs]:
            s_loc = 0.9
            matched.append(f"Location '{request.location}' covered under All-India national scheme")
        else:
            s_loc = 0.0
            unmet.append(f"Scheme restricted to specific states ({', '.join(scheme_dict.get('target_locations', []))})")
            return 0.0, matched, unmet  # Hard filter out state mismatch

        # 2. Craft Match (35%)
        if any(req_craft in c or c in req_craft for c in t_crafts if c != "all traditional crafts"):
            s_craft = 1.0
            matched.append(f"Craft technique '{request.craft_type}' specifically eligible under scheme trades")
        elif "all traditional crafts" in t_crafts or "all traditional crafts" in [c.lower() for c in t_crafts]:
            s_craft = 0.9
            matched.append(f"Craft technique '{request.craft_type}' covered under All Traditional Crafts")
        else:
            s_craft = 0.4  # General craft fallback
            unmet.append(f"Scheme specifically targets: {', '.join(scheme_dict.get('target_crafts', [])[:3])}")


        # 3. Business Status Match (20%)
        if any(req_bus in b or b in req_bus for b in e_buses):
            s_bus = 1.0
            matched.append(f"Business status '{request.business_status}' eligible")
        else:
            s_bus = 0.5
            unmet.append(f"Preferred business types: {', '.join(scheme_dict.get('eligible_business_statuses', []))}")

        # 4. Registration Status Match (10%)
        if any(req_reg in r or r in req_reg for r in e_regs):
            s_reg = 1.0
            matched.append(f"Registration status '{request.registration_status}' accepted")
        else:
            s_reg = 0.7
            unmet.append("Registration required (e.g. Pehchan / Udyam ID)")

        # 5. Demographic Filters (10%) - ONLY evaluated if scheme specifically restricts
        s_demo = 1.0
        scheme_id = scheme_dict.get("scheme_id", "")
        
        # Female-only scheme rule (e.g. Mahila Coir Yojana)
        if scheme_id == "SCHEME-MAHILA-COIR":
            if request.gender and request.gender.strip().lower() not in ["female", "woman", "women"]:
                s_demo = 0.0
                unmet.append("Mahila Coir Yojana is exclusively for female artisans")
                return 0.0, matched, unmet  # Hard filter out male for female-only scheme
            else:
                matched.append("Female artisan demographic criteria satisfied")

        # Minimum age rule (e.g. PM Vishwakarma age >= 18)
        if request.age is not None:
            if request.age < 18:
                s_demo = 0.0
                unmet.append("Minimum age requirement of 18 years not met")
                return 0.0, matched, unmet
            else:
                matched.append(f"Age ({request.age}) meets minimum eligibility requirement")

        match_score = (0.35 * s_craft) + (0.25 * s_loc) + (0.20 * s_bus) + (0.10 * s_reg) + (0.10 * s_demo)
        return round(match_score, 2), matched, unmet

    def match_schemes(self, request: SchemeMatchingRequest) -> SchemeMatchingResponse:
        """Processes SchemeMatchingRequest and returns sorted SchemeMatchingResponse."""
        self.validate_request(request)
        schemes_data = self.load_schemes_dataset()

        matched_results: List[MatchedSchemeResult] = []

        for s_dict in schemes_data:
            score, matched_criteria, unmet_criteria = self.match_scheme_for_profile(s_dict, request)

            if score < 0.20:
                continue

            scheme_item = GovernmentSchemeItem.model_validate(s_dict)
            reason = f"Matches your {request.craft_type} craft and {request.business_status} status under {request.location} coverage."

            matched_results.append(
                MatchedSchemeResult(
                    scheme=scheme_item,
                    match_score=score,
                    recommendation_reason=reason,
                    matched_criteria=matched_criteria,
                    unmet_criteria=unmet_criteria,
                )
            )

        # Sort by match_score descending
        matched_results.sort(key=lambda r: r.match_score, reverse=True)

        return SchemeMatchingResponse(
            success=True,
            total_matched=len(matched_results),
            results=matched_results,
        )
