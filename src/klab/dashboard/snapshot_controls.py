"""Shared, bounded controls for local CPU snapshot completion."""

import math

SAMPLING_DEFAULTS = {
    "sample_temperature": 0.7,
    "sample_top_k": 40,
    "sample_seed": 42,
    "sample_count": 32,
}


def validate_controls(prompt, count, temperature, top_k, seed):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise ValueError("Enter a nonempty prompt of at most 4,000 characters.")
    for value, lower, upper, label in (
        (count, 8, 512, "Output length"),
        (top_k, 0, 200, "Top-k"),
        (seed, 0, 2147483647, "Seed"),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{label} is outside the supported range.")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ValueError("Temperature must be finite and between 0 and 2.")
