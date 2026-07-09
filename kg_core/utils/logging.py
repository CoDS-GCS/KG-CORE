"""
utils/logging.py — Performance logging for KG-CORE pipeline steps.
"""

import time
import resource


def log_performance(step_name: str, start_time: float) -> None:
    """Log elapsed wall-clock time and peak RSS memory since *start_time*."""
    elapsed = time.time() - start_time
    max_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    print(f"\n=== [{step_name}] ===")
    print(f"  Time elapsed : {elapsed:.2f}s")
    print(f"  Peak RAM     : {max_rss_gb:.2f} GB")
    print("=" * (len(step_name) + 8) + "\n")
