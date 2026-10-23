"""Green Shaping: energy-accounted metadata protection for LLM-agent-orchestrated federated analytics.

Reference implementation of the paper's method (Sec. 4), analysis helpers (Sec. 5) and
evaluation protocol (Sec. 6). No paper-level results are produced by this package unless
real benchmark data are supplied and their provenance is recorded.
"""

__version__ = "0.5.0"

SYNTHETIC_LABEL = "synthetic smoke-test output, not a paper result"
REHEARSAL_LABEL = "single-host or pipeline rehearsal, not a paper result"
REAL_LABEL = "real-data output; input provenance in run_metadata.json"
