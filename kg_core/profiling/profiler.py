"""
profiling/profiler.py — C1: KG Profiler

Performs a single-pass statistical analysis of the full training graph and
returns a KGProfile shared by all downstream components:

  filtering/motif_denoising.py   — literal_scores, numeric_scores
  filtering/predicate_oracle.py  — predicate_counts (prompt construction)
  construction/sampler.py        — degree, label_counts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

import pandas as pd


@dataclass
class KGProfile:
    """
    Statistical summary of a training knowledge graph.

    Attributes
    ----------
    predicate_counts : pd.Series
        train['p'].value_counts() — predicate frequencies, descending.
        Passed to the LLM oracle prompt so the model understands both the
        vocabulary and usage prior of each relation.
    literal_scores : pd.Series
        Per-predicate ratio of unique objects to total objects.
        High scores (→1.0) indicate attribute-like / high-cardinality predicates.
    numeric_scores : pd.Series
        Per-predicate fraction of object values parseable as numbers.
    degree : pd.Series
        Combined subject + object occurrence count per entity node.
    label_counts : dict[str, pd.Series]
        Per-inference-predicate tail-entity value counts (used by the sampler).
    """
    predicate_counts: pd.Series
    literal_scores: pd.Series
    numeric_scores: pd.Series
    degree: pd.Series
    label_counts: dict = field(default_factory=dict)


def build_profile(train: pd.DataFrame, inf_predicates: Set[str]) -> KGProfile:
    """
    Compute the full statistical profile of *train*.

    Parameters
    ----------
    train : pd.DataFrame
        Full training triples with columns ['s', 'p', 'o'].
    inf_predicates : set[str]
        Predicates present in test queries — used to pre-compute per-predicate
        tail distributions for the stratified sampler.

    Returns
    -------
    KGProfile
    """
    print("[ C1 · KG Profiler ] Computing graph statistics...")

    # Predicate frequency distribution — supplied to the LLM oracle prompt
    predicate_counts = train["p"].value_counts()

    # Literal score: high uniqueness ratio → attribute-valued predicate
    pred_stats = train.groupby("p").agg(
        total=("o", "size"),
        unique_obj=("o", "nunique"),
    )
    literal_scores = pred_stats["unique_obj"] / pred_stats["total"]

    # Numeric score: high numeric prevalence → numeric-valued predicate
    numeric_scores = (
        train.groupby("p")["o"]
        .apply(lambda x: pd.to_numeric(x, errors="coerce").notna().mean())
    )

    # Entity degree: sum of subject and object occurrences per node
    degree = (
        train["s"].value_counts()
        .add(train["o"].value_counts(), fill_value=0)
    )

    # Per-inference-predicate tail distributions (consumed by the sampler)
    label_counts = {
        p: train.loc[train["p"] == p, "o"].value_counts()
        for p in inf_predicates
        if not train[train["p"] == p].empty
    }

    print(
        f"  Predicates : {len(predicate_counts):,}  |  "
        f"Entities   : {len(degree):,}  |  "
        f"Triples    : {len(train):,}"
    )

    return KGProfile(
        predicate_counts=predicate_counts,
        literal_scores=literal_scores,
        numeric_scores=numeric_scores,
        degree=degree,
        label_counts=label_counts,
    )
