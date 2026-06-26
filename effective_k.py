import math
import statistics
from collections import Counter


def summarize_effective_k(counts, requested_k=None):
    """Summarize post-enforcement cluster counts with deterministic mode ties."""
    usable = []
    excluded = 0
    for value in counts or []:
        if value is None:
            excluded += 1
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            excluded += 1
            continue
        if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
            excluded += 1
            continue
        usable.append(int(numeric))

    requested = int(requested_k) if requested_k is not None else None
    rule = "frequency_then_requested_distance_then_smaller_k"
    if not usable:
        return {
            "selected_k": None,
            "requested_k": requested,
            "n_bootstraps": 0,
            "n_excluded": int(excluded),
            "n_total": int(excluded),
            "counts": {},
            "support": None,
            "retention_rate": None,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "entropy": None,
            "normalized_entropy": None,
            "tied_modes": [],
            "tie_break_rule": rule,
        }

    frequencies = Counter(usable)
    max_frequency = max(frequencies.values())
    tied_modes = sorted(k for k, frequency in frequencies.items() if frequency == max_frequency)
    selected = (
        tied_modes[0]
        if requested is None
        else min(tied_modes, key=lambda k: (abs(k - requested), k))
    )
    probabilities = [frequency / len(usable) for frequency in frequencies.values()]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    normalized_entropy = entropy / math.log(len(frequencies)) if len(frequencies) > 1 else 0.0
    return {
        "selected_k": int(selected),
        "requested_k": requested,
        "n_bootstraps": int(len(usable)),
        "n_excluded": int(excluded),
        "n_total": int(len(usable) + excluded),
        "counts": {int(k): int(v) for k, v in sorted(frequencies.items())},
        "support": float(max_frequency / len(usable)),
        "retention_rate": (
            float(frequencies.get(requested, 0) / len(usable)) if requested is not None else None
        ),
        "mean": float(statistics.mean(usable)),
        "median": float(statistics.median(usable)),
        "min": int(min(usable)),
        "max": int(max(usable)),
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy),
        "tied_modes": [int(k) for k in tied_modes],
        "tie_break_rule": rule,
    }


def resolve_min_cluster_n(final_min_cluster_n, current_n, reference_n, mode="fixed"):
    """Resolve the operational minimum cluster size for a fit or resample."""
    final_n = int(final_min_cluster_n)
    current_n = int(current_n)
    reference_n = int(reference_n)
    mode = str(mode).strip().lower()
    if final_n < 1:
        raise ValueError("final_min_cluster_n must be positive")
    if current_n < 1 or reference_n < 1:
        raise ValueError("current_n and reference_n must be positive")
    if mode == "fixed":
        return final_n
    if mode == "scaled":
        return max(2, int(math.ceil(final_n * current_n / reference_n)))
    raise ValueError("mode must be 'fixed' or 'scaled'")
