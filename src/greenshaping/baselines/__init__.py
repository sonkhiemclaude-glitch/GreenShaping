"""Defences compared in Sec. 6.3: real baselines, adapter for external NetShaper, dummy for tests."""
from .base import Defence
from .defences import ConfigDefence, FixedPaddingDefence, NoShaping, dp_only, padding_only, rounds_only
from .dummy import DummyDefence
from .netshaper import NetShaperCaptureAdapter

__all__ = ["Defence", "ConfigDefence", "FixedPaddingDefence", "NoShaping", "DummyDefence",
           "NetShaperCaptureAdapter", "dp_only", "padding_only", "rounds_only"]
