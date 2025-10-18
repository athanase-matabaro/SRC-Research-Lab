"""
Phase B1 Runtime Guardrails Package

This package provides runtime monitoring and variance detection for compression benchmarks.

Modules:
    guardrails: GuardrailManager with variance gates, drift detection, and rollback system

Author: Athanase Nshombo (Matabaro)
Phase: B1 - Runtime Guardrails & Variance Gate
"""

from .guardrails import GuardrailManager, BaselineStats, GuardrailState

__all__ = ['GuardrailManager', 'BaselineStats', 'GuardrailState']
__version__ = '0.4.4.1'
