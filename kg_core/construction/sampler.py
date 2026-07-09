"""
construction/sampler.py — C4: Stratified Representative Sampler

Selects balanced in-context example triples for each inference predicate.

β-smoothed label distribution
    label_counts raised to power β before normalisation.
    β=0 → uniform allocation   |   β=1 → fully proportional.

Hub-avoidance weighting (α)
    Subject nodes sampled with probability ∝ 1 / (degree + 1)^α.
    Higher α gives stronger preference for low-degree nodes.

Per-label guards
    min_per_label ensures every tail class has at least a floor count.
    max_label_ratio caps over-represented classes.
"""

from __future__ import annotations

from typing import Set

import pandas as pd
from tqdm import tqdm

from ..profiling.profiler import KGProfile


def stratified_sample(
    train: pd.DataFrame,
    inf_predicates: Set[str],
    profile: KGProfile,
    beta: float,
    degree_alpha: float,
    min_per_label: int,
    max_label_ratio: float,
) -> tuple[pd.DataFrame, Set[str]]:
    """
    Perform β-smoothed, degree-aware sampling for all inference predicates.

    Returns
    -------
    sampled_df : pd.DataFrame
        Deduplicated representative triples.
    seed_entities : set[str]
        Entities in sampled_df — used as BFS seeds in the context expander.
    """
    print("[ C4 · Stratified Sampler ] Sampling representative examples...")

    sampled_rows = []
    seed_entities: Set[str] = set()

    for predicate in inf_predicates:
        relevant = train[train["p"] == predicate]
        if relevant.empty:
            print(f"  ⚠  No training triples for '{predicate}' — skipping.")
            continue

        label_counts = profile.label_counts.get(predicate, relevant["o"].value_counts())
        n_labels     = len(label_counts)
        target_total = n_labels * min_per_label
        upper        = (target_total / n_labels) * max_label_ratio

        smoothed = (label_counts + 1) ** beta
        smoothed /= smoothed.sum()

        samples_per_label = (
            (smoothed * target_total)
            .clip(lower=min_per_label, upper=upper)
            .round()
            .astype(int)
        )
        print(
            f"  {predicate}: {n_labels} labels | "
            f"target {target_total} | bounds [{min_per_label}, {upper:.0f}]"
        )

        for label, group in tqdm(
            relevant.groupby("o"),
            desc=f"  Sampling '{predicate}'",
            leave=False,
        ):
            if label not in samples_per_label:
                continue
            n = max(1, min(samples_per_label[label], len(group)))
            weights = (
                1.0 / ((profile.degree[group["s"]].values + 1) ** degree_alpha)
                if degree_alpha > 0 else None
            )
            sampled_rows.append(group.sample(n=n, weights=weights, random_state=42))

    if sampled_rows:
        sampled_df = pd.concat(sampled_rows).drop_duplicates()
        seed_entities.update(sampled_df["s"].unique())
        seed_entities.update(sampled_df["o"].unique())
        print(
            f"  Sampled {len(sampled_df):,} triples | "
            f"{len(seed_entities):,} seed entities"
        )
    else:
        sampled_df = pd.DataFrame(columns=["s", "p", "o"])
        print("  No triples sampled.")

    return sampled_df, seed_entities
