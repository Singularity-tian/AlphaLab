"""Verification of anonymization quality."""

from __future__ import annotations

import logging

from alphalab.anonymizer.pipeline import AnonymizationResult

logger = logging.getLogger(__name__)


class AnonymizationVerifier:
    """Checks that anonymization was effective."""

    def verify(
        self,
        original_text: str,
        result: AnonymizationResult,
    ) -> tuple[bool, list[str]]:
        """Returns (passed, issues)."""
        issues: list[str] = []

        # Check for residual entity leakage
        for original in result.entity_map:
            if len(original) < 3:
                continue
            if original.lower() in result.anonymized_text.lower():
                issues.append(f"Entity leakage: '{original}' still present")

        # Check length ratio
        if len(result.anonymized_text) < 100:
            issues.append("Anonymized text is suspiciously short")

        ratio = len(result.anonymized_text) / max(len(original_text), 1)
        if ratio < 0.5 or ratio > 1.5:
            issues.append(f"Suspicious length change: {ratio:.2f}x")

        if issues:
            logger.warning("Anonymization verification failed: %s", issues)

        return len(issues) == 0, issues
