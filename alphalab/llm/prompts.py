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

# ---------------------------------------------------------------------------
# Business Resilience factor — two-stage pipeline
# ---------------------------------------------------------------------------

RESILIENCE_SUMMARIZER_SYSTEM = """\
You are a financial research assistant preparing an anonymized company briefing.
Your job is to take raw company data and produce a clean, objective, structured \
summary with ALL identifying information removed."""

RESILIENCE_SUMMARIZER_USER = """\
Below is raw data about a company. Create an anonymized business briefing.

Instructions:

1. ANONYMIZE: Replace the company name, ticker symbol, product names, executive \
names, and any phrases that could identify the company (e.g., "world's most \
valuable", "Cupertino-based", unique product names) with generic labels \
(COMPANY_A, PRODUCT_A, PERSON_A, etc.). Be thorough — even indirect clues \
must be removed.

2. ORGANIZE into these sections:
   - Business Model: What the company does, how it makes money, key competitive \
advantages or disadvantages
   - Revenue Composition: Product/segment mix and geographic distribution (keep \
percentages, anonymize segment names if they are identifying)
   - Workforce & Scale: Employee count, implied labor intensity, organizational scale
   - Financial Health: Key ratios and what they indicate
   - Strategic Outlook: Management's forward-looking comments on growth, threats, \
technology adoption (especially AI/automation), and competitive dynamics \
(extract from earnings transcript if available)

3. KEEP all financial numbers, percentages, and ratios — they are not identifying \
on their own without the company name.

4. REMOVE any subjective spin. Present only objective facts and direct quotes \
(anonymized) from management.

5. Aim for 800-1200 words. Be concise but comprehensive.

Raw data:
---
{raw_data}
---

Output the anonymized briefing as plain text with clear section headers."""

RESILIENCE_JUDGE_SYSTEM = """\
You are a business analyst and technology strategist. You evaluate companies \
purely on their fundamental quality and resilience to technological disruption. \
The company identity has been anonymized — do not attempt to identify it."""

RESILIENCE_JUDGE_USER = """\
Analyze this anonymized company briefing and score on 4 dimensions (1-10 each).

{briefing}

Dimensions:

1. business_quality (1-10): Competitive moat strength — brand power, network \
effects, switching costs, intellectual property, regulatory advantages, barriers \
to entry. 10 = near-impenetrable moat. 1 = commodity business, no differentiation.

2. revenue_durability (1-10): How resilient and recurring the revenue streams are. \
Consider: subscription vs one-time, product/customer/geographic diversification, \
switching costs. 10 = highly diversified recurring revenue. 1 = single-source \
dependency.

3. ai_disruption_risk (1-10): Resilience to disruption from AI, autonomous agents, \
and advanced automation. Can AI replace the core product/service? Is the business \
labor-intensive in ways AI can automate? Does AI enhance or threaten the business \
model? 10 = benefits from AI / impossible to automate. 1 = core business easily \
replaced by AI.

4. fundamental_strength (1-10): Balance sheet health, profitability, cash \
generation, capital efficiency. 10 = exceptional margins and cash flow with \
fortress balance sheet. 1 = cash-burning with deteriorating fundamentals.

Provide a brief explanation for each score.

Return as JSON with fields: business_quality, revenue_durability, \
ai_disruption_risk, fundamental_strength, explanations (a dict mapping each \
dimension name to its explanation string)."""
