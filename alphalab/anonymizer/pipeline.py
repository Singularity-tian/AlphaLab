"""Anonymization pipeline for financial text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AnonymizationResult:
    """Result of anonymizing a text."""

    anonymized_text: str
    entity_map: dict[str, str] = field(default_factory=dict)
    reverse_map: dict[str, str] = field(default_factory=dict)


class AnonymizationPipeline:
    """Anonymizes financial text to remove entity-identifying information.

    Three layers:
    1. Direct identifiers: company name, ticker, named people
    2. Temporal identifiers: specific years, quarters
    3. Additional entities: product names, competitor references
    """

    COMPANY_PLACEHOLDER = "COMPANY_A"
    PERSON_PREFIX = "PERSON_"
    PRODUCT_PREFIX = "PRODUCT_"

    def anonymize(
        self,
        text: str,
        company_name: str,
        ticker: str,
        additional_entities: dict[str, str] | None = None,
    ) -> AnonymizationResult:
        entity_map: dict[str, str] = {}
        result_text = text

        # Layer 1: Direct identifiers
        for variation in self._get_company_variations(company_name):
            if variation in result_text:
                entity_map[variation] = self.COMPANY_PLACEHOLDER
                result_text = result_text.replace(variation, self.COMPANY_PLACEHOLDER)

        ticker_pattern = re.compile(rf"\b{re.escape(ticker)}\b", re.IGNORECASE)
        result_text = ticker_pattern.sub(self.COMPANY_PLACEHOLDER, result_text)
        entity_map[ticker] = self.COMPANY_PLACEHOLDER

        # Layer 2: Temporal identifiers
        year_pattern = re.compile(r"\b(20\d{2})\b")
        years_found = sorted(set(year_pattern.findall(result_text)))
        for i, year in enumerate(years_found):
            offset = i - len(years_found) + 1
            placeholder = f"YEAR_T{'+' if offset > 0 else ''}{offset}"
            entity_map[year] = placeholder
            result_text = result_text.replace(year, placeholder)

        quarter_pattern = re.compile(r"\b(Q[1-4])\b")
        result_text = quarter_pattern.sub("QUARTER_N", result_text)

        # Layer 3: Additional entities
        if additional_entities:
            person_count = 0
            product_count = 0
            for entity, category in additional_entities.items():
                if category == "person":
                    placeholder = f"{self.PERSON_PREFIX}{chr(65 + person_count)}"
                    person_count += 1
                elif category == "product":
                    placeholder = f"{self.PRODUCT_PREFIX}{chr(65 + product_count)}"
                    product_count += 1
                else:
                    continue

                entity_map[entity] = placeholder
                result_text = re.sub(
                    re.escape(entity), placeholder, result_text, flags=re.IGNORECASE
                )

        reverse_map = {v: k for k, v in entity_map.items()}

        return AnonymizationResult(
            anonymized_text=result_text,
            entity_map=entity_map,
            reverse_map=reverse_map,
        )

    def _get_company_variations(self, company_name: str) -> list[str]:
        variations = [company_name]
        for suffix in [
            " Inc.", " Inc", " Corp.", " Corp", " Ltd.", " Ltd",
            " LLC", " Co.", " Company",
        ]:
            stripped = company_name.replace(suffix, "").strip()
            if stripped != company_name:
                variations.append(stripped)
        variations.append(f"{company_name}'s")
        if len(variations) > 1:
            variations.append(f"{variations[1]}'s")
        # Longest first to avoid partial replacements
        variations.sort(key=len, reverse=True)
        return variations
