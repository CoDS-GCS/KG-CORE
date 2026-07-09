"""
construction/expander.py — C5: Direction-Aware Context Expander

Extracts the local structural neighborhood of seed entities via BFS over
the oracle-filtered, non-motif-cleaned graph.

Direction parameter
-------------------
outgoing  — subject → object edges.  Captures causal / mechanistic paths
            leading FROM seed entities.
incoming  — object → subject edges.  Captures what entities point TO seeds.
both      — bidirectional; maximum structural coverage.

Seeds come from two sources (dual seeding):
  1. Query head entities h from the test set (always present).
  2. Entities from the stratified sample (C4), expanding context around
     representative examples.
"""

from __future__ import annotations

from typing import Optional, Set

import pandas as pd


def expand_context(
    train: pd.DataFrame,
    seed_entities: Set[str],
    non_motif_predicates: Set[str],
    llm_predicates: Optional[Set[str]],
    num_hops: int,
    direction: str,
) -> pd.DataFrame:
    """
    BFS expansion from *seed_entities* over the filtered training graph.

    Parameters
    ----------
    train : pd.DataFrame
        Full training graph ['s', 'p', 'o'].
    seed_entities : set[str]
        Initial BFS frontier — query heads + sampled example entities.
    non_motif_predicates : set[str]
        Predicates removed by C2.  Always excluded from traversal.
    llm_predicates : set[str] or None
        If provided, restricts BFS to oracle-selected predicates (C3).
        Pass None to disable oracle filtering.
    num_hops : int
        BFS depth.  1 → single-hop (no seed growth between iterations).
    direction : str
        One of {'outgoing', 'incoming', 'both'}.

    Returns
    -------
    pd.DataFrame
        Deduplicated context triples ['s', 'p', 'o'].
    """
    print(
        f"[ C5 · Context Expander ]  "
        f"direction={direction} | hops={num_hops} | "
        f"seeds={len(seed_entities):,} | "
        f"oracle={'on' if llm_predicates is not None else 'off'}"
    )

    context_graph = _build_context_graph(train, non_motif_predicates, llm_predicates)
    print(f"  Filtered graph  : {len(context_graph):,} triples")

    current_seeds = set(seed_entities)
    collected = []

    for hop in range(max(1, num_hops)):
        hop_df = _single_hop(context_graph, current_seeds, direction)
        if hop_df.empty:
            print(f"  Hop {hop + 1}: empty — stopping early.")
            break

        collected.append(hop_df)
        print(f"  Hop {hop + 1}: {len(hop_df):,} triples")

        if num_hops <= 1:
            break  # single-hop: do not grow the frontier

        new_entities = pd.concat([hop_df["s"], hop_df["o"]]).unique()
        current_seeds.update(new_entities)

    if not collected:
        return pd.DataFrame(columns=["s", "p", "o"])

    result = pd.concat(collected).drop_duplicates()
    print(f"  Total context   : {len(result):,} triples")
    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_context_graph(
    train: pd.DataFrame,
    non_motif_predicates: Set[str],
    llm_predicates: Optional[Set[str]],
) -> pd.DataFrame:
    """Apply C2 and C3 filters to produce the BFS search space."""
    mask = ~train["p"].isin(non_motif_predicates)
    if llm_predicates is not None:
        mask &= train["p"].isin(llm_predicates)
    return train[mask]


def _single_hop(
    context_graph: pd.DataFrame,
    seeds: Set[str],
    direction: str,
) -> pd.DataFrame:
    """Return all context-graph triples touching *seeds* in *direction*."""
    if direction == "outgoing":
        mask = context_graph["s"].isin(seeds)
    elif direction == "incoming":
        mask = context_graph["o"].isin(seeds)
    else:  # both
        mask = context_graph["s"].isin(seeds) | context_graph["o"].isin(seeds)
    return context_graph[mask]
