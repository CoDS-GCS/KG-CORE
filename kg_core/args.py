"""
args.py — CLI argument parsing for KG-CORE.

All pipeline hyperparameters and path configurations are defined here.
Default values are kept in sync with configs/default.yaml — prefer the
YAML for batch runs, and these flags for quick overrides.
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KG-CORE: Direction-aware, LLM-guided subgraph sampler for GFM tail prediction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Graph traversal ──────────────────────────────────────────────────────
    parser.add_argument("--multi_hop", action="store_true", default=True)
    parser.add_argument("--no_multi_hop", action="store_false", dest="multi_hop")
    parser.add_argument(
        "--hops", type=int, default=1,
        help="BFS expansion depth (ignored when --no_multi_hop is set)."
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["both", "incoming", "outgoing"],
        default="both",
        help="Edge traversal direction during context expansion."
    )

    # ── Oracle & sampling toggles ────────────────────────────────────────────
    parser.add_argument(
        "--use_LLM",
        action="store_true",
        help="Restrict context expansion to LLM-selected predicates."
    )
    parser.add_argument(
        "--use_sampling",
        action="store_true",
        default=True,
        help="Enable stratified representative sampling (default: True)."
    )
    parser.add_argument(
        "--no_sampling",
        action="store_false",
        dest="use_sampling",
        help="Disable sampling — produce a context-only subgraph."
    )

    # ── Filter mode (early-exit shortcuts) ───────────────────────────────────
    parser.add_argument(
        "--filter_mode",
        type=str,
        choices=["full", "literal_only", "numeric_only", "literal_and_numeric", "llm_only"],
        default="full",
        help=(
            "Pipeline execution mode:\n"
            "  full                — complete KG-CORE pipeline (default)\n"
            "  literal_only        — output after literal filtering only\n"
            "  numeric_only        — output after numeric filtering only\n"
            "  literal_and_numeric — output after both literal + numeric filtering\n"
            "  llm_only            — keep only LLM-suggested predicates"
        )
    )

    # ── Sampling hyperparameters ─────────────────────────────────────────────
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="Tail-distribution smoothing exponent (0=uniform, 1=proportional)."
    )
    parser.add_argument(
        "--degree_alpha", type=float, default=0.5,
        help="Inverse-degree weighting strength for subject node sampling."
    )
    parser.add_argument(
        "--min_per_label", type=int, default=4,
        help="Minimum examples sampled per tail-entity class."
    )
    parser.add_argument(
        "--max_label_ratio", type=float, default=100,
        help="Soft upper cap multiplier on per-label sample counts."
    )

    # ── Literal detection thresholds ─────────────────────────────────────────
    parser.add_argument(
        "--literal_threshold", type=float, default=0.9,
        help="Uniqueness ratio above which a predicate is deemed literal."
    )
    parser.add_argument(
        "--numeric_threshold", type=float, default=0.9,
        help="Numeric-object prevalence above which a predicate is flagged numeric."
    )

    # ── I/O paths ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--path_train", type=str,
        default="/shared_mnt/ULTRA/grail_s/yago3/versions/train_FG_YAGO3.txt"
    )
    parser.add_argument(
        "--path_test", type=str,
        default="/shared_mnt/ULTRA/grail_s/yago3/versions/test_playsFor.txt"
    )
    parser.add_argument(
        "--path_llm_preds", type=str,
        default="/shared_mnt/ULTRA/grail_s/yago3/versions/llm_preds_playsFor.txt"
    )
    parser.add_argument(
        "--train_replace_keyword", type=str, default="FG",
        help="Keyword in the train path to replace when naming the output file."
    )
    parser.add_argument(
        "--custom_output_postfix", type=str, default=None,
        help="Optional tag appended to the output filename."
    )

    return parser.parse_args()


def build_output_path(args: argparse.Namespace) -> str:
    """Construct the output file path from args, mirroring the input train path."""
    postfix  = f"_{args.custom_output_postfix}" if args.custom_output_postfix else ""
    pred_tag = args.path_test.split("/")[-1].split("_")[-1].replace(".txt", "")
    hop_tag  = args.hops if args.multi_hop else 1
    return args.path_train.replace(
        args.train_replace_keyword,
        f"train_{pred_tag}_dir-{args.direction}_H{hop_tag}_sampling-{args.use_sampling}{postfix}",
    )
