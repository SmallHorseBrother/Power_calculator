"""
Compatibility wrapper for README examples.

Importing `process_muscle_up_video` from this module keeps the public API
stable while the core implementation lives in `power_calculator_mu.py`.
"""

from power_calculator_mu import process_muscle_up_video

__all__ = ["process_muscle_up_video"]
