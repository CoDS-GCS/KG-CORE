"""
utils/io.py — File I/O helpers for KG-CORE.

Provides the shared triple-save routine used by the assembler and the
early-exit filter modes.
"""

import pandas as pd


def save_triples(
    df: pd.DataFrame,
    path_train: str,
    replace_keyword: str,
    suffix: str,
) -> str:
    """
    Clean *df* and write it as a tab-separated file.

    The output path is derived from *path_train* by replacing *replace_keyword*
    with *suffix*.  Raises if the resolved output path equals the input path
    (which would silently overwrite the source file).

    Returns the resolved output path.
    """
    for col in ["s", "p", "o"]:
        df[col] = df[col].astype(str).str.strip()
    df = df.replace(["", "nan", "None"], pd.NA).dropna()

    path_out = path_train.replace(replace_keyword, f"train_{suffix}")
    if path_train == path_out:
        raise ValueError(
            "Resolved output path is identical to the input path — "
            "this would overwrite the source file. "
            "Check --train_replace_keyword."
        )

    df.to_csv(path_out, header=None, index=None, sep="\t")
    return path_out
