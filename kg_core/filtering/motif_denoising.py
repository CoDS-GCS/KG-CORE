"""
filtering/motif_denoising.py — C2: Non-Motif Filtering

Identifies and removes literal- and numeric-valued predicates from the
candidate predicate set.  These predicates describe entity *attributes*
rather than entity-to-entity structural relations, and including them
in the subgraph degrades motif-based reasoning.

Detection mechanisms
--------------------
1. Uniqueness ratio  — unique_obj / total > δ_lit  →  literal predicate
2. Numeric coverage  — fraction parseable as float > δ_num  →  numeric predicate

Inference predicates (from test queries) are always excluded from removal,
since they are the target relations and must be preserved.

This module also handles the four early-exit diagnostic filter modes:
  literal_only | numeric_only | literal_and_numeric | llm_only
"""

from __future__ import annotations

from typing import Optional, Set

import pandas as pd

from ..profiling.profiler import KGProfile
from ..utils.io import save_triples


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_literal_predicates(
    profile: KGProfile,
    threshold: float,
    inf_predicates: Set[str],
) -> Set[str]:
    """Return predicates whose object uniqueness ratio exceeds *threshold*."""
    flagged = set(profile.literal_scores[profile.literal_scores > threshold].index)
    return flagged - inf_predicates


def detect_numeric_predicates(
    profile: KGProfile,
    threshold: float,
    inf_predicates: Set[str],
) -> Set[str]:
    """Return predicates where more than *threshold* of objects parse as numbers."""
    flagged = set(profile.numeric_scores[profile.numeric_scores > threshold].index)
    return flagged - inf_predicates


def get_non_motif_predicates(
    profile: KGProfile,
    literal_threshold: float,
    numeric_threshold: float,
    inf_predicates: Set[str],
) -> tuple[Set[str], Set[str]]:
    """
    Return (literal_predicates, numeric_predicates) as separate sets so
    callers can report and ablate each independently.
    """
    literal_preds = detect_literal_predicates(profile, literal_threshold, inf_predicates)
    numeric_preds = detect_numeric_predicates(profile, numeric_threshold, inf_predicates)

    print(f"[ C2 · Non-Motif Filter ]")
    print(f"  Literal predicates : {len(literal_preds)} → {literal_preds}")
    print(f"  Numeric predicates : {len(numeric_preds)} → {numeric_preds}")

    return literal_preds, numeric_preds


# ── Early-exit filter modes ───────────────────────────────────────────────────

def run_filter_mode(
    mode: str,
    train: pd.DataFrame,
    literal_preds: Set[str],
    numeric_preds: Set[str],
    inf_predicates: Set[str],
    path_train: str,
    replace_keyword: str,
    custom_postfix: str,
    llm_preds: Optional[pd.DataFrame],
    literal_threshold: float,
    numeric_threshold: float,
) -> bool:
    """
    Execute an early-exit filter mode, write the filtered graph to disk,
    and return True so the caller knows to stop the pipeline.

    Returns False when *mode* is 'full' (no early exit).
    """
    if mode == "full":
        return False

    if mode == "literal_only":
        _filter_and_report(
            train, literal_preds,
            suffix=f"filter-literal_thresh{literal_threshold}{custom_postfix}",
            path_train=path_train, replace_keyword=replace_keyword,
            label="literal_only",
        )

    elif mode == "numeric_only":
        _filter_and_report(
            train, numeric_preds,
            suffix=f"filter-numeric_thresh{numeric_threshold}{custom_postfix}",
            path_train=path_train, replace_keyword=replace_keyword,
            label="numeric_only",
        )

    elif mode == "literal_and_numeric":
        _filter_and_report(
            train, literal_preds | numeric_preds,
            suffix=f"filter-literal_and_numeric_lt{literal_threshold}_nt{numeric_threshold}{custom_postfix}",
            path_train=path_train, replace_keyword=replace_keyword,
            label="literal_and_numeric",
        )

    elif mode == "llm_only":
        if llm_preds is None:
            raise ValueError("filter_mode='llm_only' requires a valid --path_llm_preds.")
        llm_pred_set = set(llm_preds["p"].unique())
        all_preds = set(train["p"].unique())
        to_remove = all_preds - llm_pred_set - inf_predicates
        print(
            f"  LLM predicates to keep        : {len(llm_pred_set)}\n"
            f"  Inference predicates retained : {len(inf_predicates)}\n"
            f"  Predicates removed            : {len(to_remove)}"
        )
        _filter_and_report(
            train, to_remove,
            suffix=f"filter-llm_only{custom_postfix}",
            path_train=path_train, replace_keyword=replace_keyword,
            label="llm_only",
        )

    return True  # signal early exit


def _filter_and_report(
    train: pd.DataFrame,
    predicates_to_remove: Set[str],
    suffix: str,
    path_train: str,
    replace_keyword: str,
    label: str,
) -> None:
    filtered = train[~train["p"].isin(predicates_to_remove)].copy()
    path_out = save_triples(filtered, path_train, replace_keyword, suffix)
    removed = len(train) - len(filtered)
    print(
        f"[ Filter mode: {label} ]  "
        f"Kept {len(filtered):,} triples  "
        f"(removed {removed:,} — {removed / len(train) * 100:.1f}%)\n"
        f"  Written → {path_out}"
    )
