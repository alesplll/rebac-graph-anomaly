"""Anomaly patterns.

Importing this package registers every pattern. The modules below are imported
for their side effect, hence the unused-import suppressions.
"""

from rga.generator.anomalies import (  # noqa: F401
    bypass,
    collective,
    compromise,
    escalation,
    persistence,
)
from rga.generator.anomalies.base import (  # noqa: F401
    AnomalyLabel,
    AnomalyPattern,
    Injection,
    InjectionContext,
    NoCandidateError,
    available_patterns,
    get_pattern,
)
