"""
construction/assembler.py — C6: Subgraph Assembler

Merges the outputs of the Stratified Sampler (C4) and the Context Expander
(C5) into the final inference subgraph, then applies:

  1. Deduplication        — removes triple-level duplicates across sources.
  2. Leakage removal      — excludes any triple present in the test set.
                            This is a correctness guarantee: a leaked test
                            triple would constitute an oracle leak and
                            invalidate evaluation results.
  3. Type normalisation   — strips whitespace, enforces str dtype, drops
                            rows containing empty / NaN / null-like values.

The assembled subgraph is written to disk and its compression ratio relative
to the full training graph is logged.  This metric can be reported in the
system characterisation section of a paper independently of downstream GFM
performance.
"""

import pandas as pd

from ..utils.io import save_triples


def assemble(
    sampled_df: pd.DataFrame,
    context_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_size: int,
    path_train: str,
    replace_keyword: str,
    output_suffix: str,
    use_llm: bool,
) -> pd.DataFrame:
    """
    Merge, clean, and persist the final inference subgraph.

    Parameters
    ----------
    sampled_df : pd.DataFrame   Output of the Stratified Sampler (C4).
    context_df : pd.DataFrame   Output of the Context Expander (C5).
    test_df : pd.DataFrame      Test queries — used for leakage detection.
    train_size : int            Triple count of the full training graph.
    path_train : str            Base path for output filename derivation.
    replace_keyword : str       Keyword replaced in *path_train*.
    output_suffix : str         Replacement string for the output filename.
    use_llm : bool              Logged for experiment tracking.

    Returns
    -------
    pd.DataFrame
        The cleaned final subgraph (also written to disk).
    """
    print("[ C6 · Subgraph Assembler ] Merging and cleaning...")

    # 1. Merge
    merged = pd.concat([sampled_df, context_df]).drop_duplicates()
    print(f"  After merge        : {len(merged):,} triples")

    # 2. Leakage removal
    final_df = (
        pd.merge(merged, test_df, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
    )
    leaked = len(merged) - len(final_df)
    if leaked:
        print(f"  Leakage removed    : {leaked} test triple(s) excluded")

    # 3. Type normalisation
    for col in ["s", "p", "o"]:
        final_df[col] = final_df[col].astype(str).str.strip()
    final_df = final_df.replace(["", "nan", "None"], pd.NA).dropna()

    # 4. Persist
    path_out = save_triples(final_df, path_train, replace_keyword, output_suffix)

    # 5. Report
    reduction_pct = (1 - len(final_df) / train_size) * 100
    print(
        f"  Final subgraph     : {len(final_df):,} triples "
        f"({reduction_pct:.1f}% reduction from full KG)\n"
        f"  LLM oracle active  : {use_llm}\n"
        f"  Written → {path_out}"
    )
    return final_df
