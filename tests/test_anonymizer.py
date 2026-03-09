"""Tests for the anonymization pipeline."""

from alphalab.anonymizer.pipeline import AnonymizationPipeline
from alphalab.anonymizer.verification import AnonymizationVerifier


class TestAnonymizationPipeline:
    def setup_method(self):
        self.pipeline = AnonymizationPipeline()

    def test_basic_company_replacement(self):
        text = "Apple Inc. reported strong Q3 2023 earnings."
        result = self.pipeline.anonymize(
            text=text, company_name="Apple Inc.", ticker="AAPL"
        )

        assert "Apple" not in result.anonymized_text
        assert "AAPL" not in result.anonymized_text
        assert "COMPANY_A" in result.anonymized_text

    def test_ticker_replacement(self):
        text = "AAPL stock surged after the announcement."
        result = self.pipeline.anonymize(
            text=text, company_name="Apple Inc.", ticker="AAPL"
        )

        assert "AAPL" not in result.anonymized_text
        assert "COMPANY_A" in result.anonymized_text

    def test_year_replacement(self):
        text = "Revenue grew 15% in 2023 compared to 2022."
        result = self.pipeline.anonymize(
            text=text, company_name="TestCo", ticker="TST"
        )

        assert "2023" not in result.anonymized_text
        assert "2022" not in result.anonymized_text
        assert "YEAR_T" in result.anonymized_text

    def test_quarter_replacement(self):
        text = "Q3 results exceeded Q2 guidance."
        result = self.pipeline.anonymize(
            text=text, company_name="TestCo", ticker="TST"
        )

        assert "Q3" not in result.anonymized_text
        assert "QUARTER_N" in result.anonymized_text

    def test_additional_entities(self):
        text = "Tim Cook announced the new iPhone 15 lineup."
        result = self.pipeline.anonymize(
            text=text,
            company_name="Apple Inc.",
            ticker="AAPL",
            additional_entities={"Tim Cook": "person", "iPhone": "product"},
        )

        assert "Tim Cook" not in result.anonymized_text
        assert "iPhone" not in result.anonymized_text
        assert "PERSON_A" in result.anonymized_text
        assert "PRODUCT_A" in result.anonymized_text

    def test_entity_map_populated(self):
        text = "Apple reported earnings."
        result = self.pipeline.anonymize(
            text=text, company_name="Apple", ticker="AAPL"
        )

        assert "Apple" in result.entity_map or "AAPL" in result.entity_map
        assert len(result.reverse_map) > 0

    def test_company_variations(self):
        variations = self.pipeline._get_company_variations("Apple Inc.")
        assert "Apple Inc." in variations
        assert "Apple" in variations
        # Longest first
        assert len(variations[0]) >= len(variations[-1])


class TestAnonymizationVerifier:
    def test_passes_clean_anonymization(self):
        from alphalab.anonymizer.pipeline import AnonymizationResult

        original = (
            "Apple reported strong results in Q3. Revenue grew significantly "
            "across all segments, with services reaching new highs. Management "
            "expressed confidence in the product pipeline and forward guidance."
        )
        anon = (
            "COMPANY_A reported strong results in QUARTER_N. Revenue grew significantly "
            "across all segments, with services reaching new highs. Management "
            "expressed confidence in the product pipeline and forward guidance."
        )
        result = AnonymizationResult(
            anonymized_text=anon,
            entity_map={"Apple": "COMPANY_A"},
            reverse_map={"COMPANY_A": "Apple"},
        )

        verifier = AnonymizationVerifier()
        passed, issues = verifier.verify(original, result)

        assert passed
        assert len(issues) == 0

    def test_detects_entity_leakage(self):
        from alphalab.anonymizer.pipeline import AnonymizationResult

        result = AnonymizationResult(
            anonymized_text="Apple reported strong results.",
            entity_map={"Apple": "COMPANY_A"},
            reverse_map={"COMPANY_A": "Apple"},
        )

        verifier = AnonymizationVerifier()
        passed, issues = verifier.verify("Apple reported strong results.", result)

        assert not passed
        assert any("leakage" in i.lower() for i in issues)
