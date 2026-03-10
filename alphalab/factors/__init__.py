"""Factor modules — import to register shared factors.

Strategy-specific factors are registered by their own entry points
(e.g., strategies/graham_value/run.py imports strategies.graham_value.factors).
"""

import alphalab.factors.business_resilience  # noqa: F401
import alphalab.factors.momentum  # noqa: F401
import alphalab.factors.traditional  # noqa: F401
