"""
filtering/predicate_oracle.py — C3: LLM Predicate Oracle

Identifies task-relevant predicates for a given inference relation.

Two modes are supported:

  Static (current default)
  ─────────────────────────
  Loads a pre-computed predicate list from a flat file (one predicate per
  line).  Mirrors the original llm_preds.txt design.

  Live (recommended for publication)
  ────────────────────────────────────
  Prompts an LLM with the KG predicate frequency distribution and the target
  inference predicate r, returning a ranked list of structurally and
  semantically relevant predicates to retain during context expansion.

  This transforms a static design decision into a tunable, task-conditioned
  component — the key publishable novelty of KG-CORE.

Usage
-----
  oracle = LLMPredicateOracle.from_file(path_llm_preds)
  # or
  oracle = LLMPredicateOracle.from_llm(profile, test_predicate, top_k=20)

  filtered_graph = oracle.filter_graph(train, struct_predicates)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from ..profiling.profiler import KGProfile


class LLMPredicateOracle:
    """
    Wraps the set of task-relevant predicates selected by the oracle.

    Attributes
    ----------
    predicates : set[str]
        The predicates deemed relevant for the inference task.
    source : str
        How the predicates were obtained ('file' or 'llm').
    """

    def __init__(self, predicates: Set[str], source: str = "unknown") -> None:
        self.predicates = predicates
        self.source = source

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str) -> "LLMPredicateOracle":
        """Load predicates from a pre-computed flat file (tab-separated, column 'p')."""
        df = pd.read_csv(path, header=None, sep="\t", names=["p"], dtype=str)
        predicates = set(df["p"].dropna().str.strip().unique())
        print(
            f"[ C3 · LLM Oracle ] Loaded {len(predicates)} predicates "
            f"from: {Path(path).name}"
        )
        return cls(predicates=predicates, source="file")

    @classmethod
    def from_llm(
        cls,
        profile: KGProfile,
        test_predicate: str,
        top_k: int = 20,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ) -> "LLMPredicateOracle":
        """
        Query an LLM at runtime to select task-relevant predicates.

        The prompt supplies:
          - The full predicate frequency distribution (predicate_counts)
            so the model understands the KG vocabulary and usage prior.
          - The target inference predicate r so the model can reason about
            which structural context is relevant for predicting its missing tails.

        Parameters
        ----------
        profile : KGProfile
            Output of the KG Profiler — provides predicate_counts.
        test_predicate : str
            The relation r from test queries (h, r, ?).
        top_k : int
            Number of predicates to request; acts as a tunable hyperparameter.
        model : str
            LLM identifier (OpenAI model string or compatible endpoint).
        api_key : str, optional
            API key.  Falls back to the OPENAI_API_KEY environment variable.
        """
        prompt = _build_oracle_prompt(profile.predicate_counts, test_predicate, top_k)
        predicates = _call_llm(prompt, model=model, api_key=api_key)
        print(
            f"[ C3 · LLM Oracle ] Retrieved {len(predicates)} predicates "
            f"from {model} for predicate '{test_predicate}' (top_k={top_k})"
        )
        return cls(predicates=predicates, source="llm")

    # ── Filtering helper ──────────────────────────────────────────────────────

    def filter_graph(
        self,
        graph: pd.DataFrame,
        struct_predicates: Set[str],
    ) -> pd.DataFrame:
        """
        Return the subgraph restricted to oracle-selected predicates,
        further intersected with the non-motif-filtered predicate set (C2).
        """
        allowed = self.predicates & struct_predicates
        filtered = graph[graph["p"].isin(allowed)]
        print(
            f"[ C3 · LLM Oracle ] Graph filtered: "
            f"{len(filtered):,} / {len(graph):,} triples retained "
            f"({len(allowed)} oracle-allowed predicates)"
        )
        return filtered


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_oracle_prompt(
    predicate_counts: pd.Series,
    test_predicate: str,
    top_k: int,
) -> str:
    """
    Build the oracle prompt from predicate statistics and the target relation.

    The frequency distribution gives the model both a vocabulary and a prior
    on how common each relation is — context needed to reason about which
    predicates provide meaningful structural signal for predicting *test_predicate*.
    """
    pred_json = json.dumps(predicate_counts.to_dict(), indent=2)
    return (
        f"You are an expert in knowledge graph reasoning.\n\n"
        f"A knowledge graph contains the following predicates and their "
        f"occurrence frequencies:\n{pred_json}\n\n"
        f"The inference task is tail prediction for the relation: '{test_predicate}'\n"
        f"That is, given a query (head_entity, '{test_predicate}', ?), "
        f"we want to predict the missing tail entity.\n\n"
        f"Which predicates carry structural or semantic information most useful "
        f"for predicting the missing tail?\n\n"
        f"Return ONLY a JSON array of the top {top_k} most relevant predicate "
        f"strings, ordered by relevance, with no explanation or extra text.\n"
        f'Example format: ["predicate_a", "predicate_b", ...]'
    )


def _call_llm(prompt: str, model: str, api_key: Optional[str]) -> Set[str]:
    """Send *prompt* to the specified LLM and parse the returned predicate list."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "Live LLM oracle requires the 'openai' package: pip install openai"
        ) from exc

    import os
    client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON array.")
        return set(str(p).strip() for p in parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"LLM oracle returned an unparseable response:\n{raw}"
        ) from exc
