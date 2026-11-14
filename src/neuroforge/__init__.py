"""NeuroForge study contracts and reproducibility helpers."""

from .contracts import (
    AcceptanceConfig,
    CandidateRecord,
    CloudPreflight,
    CostEvent,
    DataItem,
    DiagnosticFeatures,
    FaultConfig,
    GateDecision,
    GateOutcome,
    Metrics,
    PatchManifest,
    RunConfig,
    StageResult,
    canonical_json,
    config_hash,
    load_config,
)

__all__ = [
    "AcceptanceConfig",
    "CandidateRecord",
    "CloudPreflight",
    "CostEvent",
    "DataItem",
    "DiagnosticFeatures",
    "FaultConfig",
    "GateDecision",
    "GateOutcome",
    "Metrics",
    "PatchManifest",
    "RunConfig",
    "StageResult",
    "canonical_json",
    "config_hash",
    "load_config",
]

__version__ = "0.1.0"
