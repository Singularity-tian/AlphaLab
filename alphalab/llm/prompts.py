"""Prompt templates for LLM-powered factors."""

EARNINGS_SENTIMENT_SYSTEM = """\
You are a financial analyst evaluating an earnings call transcript.
The company identity has been anonymized. Evaluate the transcript purely on its content.
Do not attempt to identify the company. Focus on:
1. Revenue and earnings trajectory
2. Management tone and confidence
3. Forward guidance strength
4. Margin trends
5. Risk factors mentioned"""

EARNINGS_SENTIMENT_USER = """\
Analyze the following anonymized earnings call transcript and provide
a sentiment assessment.

Transcript:
---
{transcript}
---

Evaluate on the following dimensions and provide scores from 1 (very negative) \
to 10 (very positive):
1. overall_sentiment: General tone and outlook
2. revenue_outlook: Revenue growth trajectory
3. margin_trend: Profitability direction
4. guidance_strength: Confidence in forward guidance
5. risk_level: Level of risks mentioned (10 = low risk, 1 = high risk)

Also provide a brief explanation for each score."""
