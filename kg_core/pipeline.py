"""
pipeline.py — KG-CORE end-to-end pipeline orchestrator

Wires the six components in sequence:

  C1  profiling/profiler.py            → KGProfile
  C2  filtering/motif_denoising.py     → non_motif_preds  (early exit if filter_mode != 'full')
  C3  filtering/predicate_oracle.py    → LLMPredicateOracle
  C4  construction/sampler.py          → sampled_df, seed_entities
  C5  construction/expander.py         → context_df
  C6  construction/assembler.py        → final subgraph (written to disk)

Each component lives in its own module.  To swap or extend a component,
replace its import and call — the rest of the pipeline is unaffected.
"""

import time

import pandas as pd

from .args import build_output_path
from .profiling.profiler import build_profile
from .filtering.motif_denoising import get_non_motif_predicates, run_filter_mode
from .filtering.predicate_oracle import LLMPredicateOracle
from .construction.sampler import stratified_sample
from .construction.expander import expand_context
from .construction.assembler import assemble
from .utils.logging import log_performance


def run(args) -> None:
    """Execute the full KG-CORE pipeline from parsed *args*."""

    start_time = time.time()

    path_train      = args.path_train
    path_test       = args.path_test
    path_llm_preds  = args.path_llm_preds
    replace_keyword = args.train_replace_keyword
    use_llm         = args.use_LLM
    filter_mode     = args.filter_mode
    custom_postfix  = f"_{args.custom_output_postfix}" if args.custom_output_postfix else ""

    # ── Load raw data ─────────────────────────────────────────────────────────
    print(f"\nLoading data for: {path_test}")
    train = pd.read_csv(path_train, header=None, sep="\t", names=["s", "p", "o"], dtype=str)
    test  = pd.read_csv(path_test,  header=None, sep="\t", names=["s", "p", "o"], dtype=str)

    llm_preds_df = None
    if use_llm or filter_mode == "llm_only":
        llm_preds_df = pd.read_csv(
            path_llm_preds, header=None, sep="\t", names=["p"], dtype=str
        )

    inf_predicates = set(test["p"].unique())
    print(f"  Inference predicates : {inf_predicates}")

    # ── C1  KG Profiler ───────────────────────────────────────────────────────
    profile = build_profile(train, inf_predicates)
    log_performance("C1 · KG Profiler", start_time)

    # ── C2  Non-Motif Filtering ───────────────────────────────────────────────
    literal_preds, numeric_preds = get_non_motif_predicates(
        profile,
        literal_threshold=args.literal_threshold,
        numeric_threshold=args.numeric_threshold,
        inf_predicates=inf_predicates,
    )

    early_exit = run_filter_mode(
        mode=filter_mode,
        train=train,
        literal_preds=literal_preds,
        numeric_preds=numeric_preds,
        inf_predicates=inf_predicates,
        path_train=path_train,
        replace_keyword=replace_keyword,
        custom_postfix=custom_postfix,
        llm_preds=llm_preds_df,
        literal_threshold=args.literal_threshold,
        numeric_threshold=args.numeric_threshold,
    )
    if early_exit:
        log_performance("C2 · Non-Motif Filter (early exit)", start_time)
        return

    non_motif_preds = literal_preds | numeric_preds
    print(f"\n  Total non-motif predicates excluded: {len(non_motif_preds)}")
    log_performance("C2 · Non-Motif Filter", start_time)

    # ── C3  LLM Predicate Oracle ──────────────────────────────────────────────
    oracle_preds = None
    if use_llm:
        # Static file mode (current default).
        # Swap to LLMPredicateOracle.from_llm(...) for live querying.
        oracle = LLMPredicateOracle.from_file(path_llm_preds)
        oracle_preds = oracle.predicates
        log_performance("C3 · LLM Predicate Oracle", start_time)

    # ── C4  Stratified Sampler ────────────────────────────────────────────────
    seed_entities = set(test["s"].unique())  # always seed from query heads

    if args.use_sampling:
        sampled_df, example_seeds = stratified_sample(
            train=train,
            inf_predicates=inf_predicates,
            profile=profile,
            beta=args.beta,
            degree_alpha=args.degree_alpha,
            min_per_label=args.min_per_label,
            max_label_ratio=args.max_label_ratio,
        )
        seed_entities.update(example_seeds)
    else:
        print("[ C4 · Stratified Sampler ] DISABLED — context-only mode.")
        sampled_df = pd.DataFrame(columns=["s", "p", "o"])

    log_performance("C4 · Stratified Sampler", start_time)

    # ── C5  Context Expander ──────────────────────────────────────────────────
    context_df = expand_context(
        train=train,
        seed_entities=seed_entities,
        non_motif_predicates=non_motif_preds,
        llm_predicates=oracle_preds,
        num_hops=args.hops if args.multi_hop else 1,
        direction=args.direction,
    )
    log_performance("C5 · Context Expander", start_time)

    # ── C6  Subgraph Assembler ────────────────────────────────────────────────
    output_path = build_output_path(args)
    if path_train == output_path:
        raise RuntimeError(
            "Resolved output path matches the input — would overwrite the source file. "
            "Check --train_replace_keyword."
        )
    output_suffix = output_path.split("/")[-1].replace("train_", "", 1).replace(".txt", "")

    assemble(
        sampled_df=sampled_df,
        context_df=context_df,
        test_df=test,
        train_size=len(train),
        path_train=path_train,
        replace_keyword=replace_keyword,
        output_suffix=output_suffix,
        use_llm=use_llm,
    )
    log_performance("C6 · Subgraph Assembler", start_time)
    log_performance("Pipeline complete", start_time)
