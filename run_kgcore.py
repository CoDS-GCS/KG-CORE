"""
run_kgcore.py — KG-CORE entry point

Usage
-----
  # Full pipeline with LLM oracle (static predicate file)
  python run_kgcore.py \
      --path_train /data/train_FG.txt \
      --path_test  /data/test.txt \
      --path_llm_preds /data/llm_preds.txt \
      --use_LLM --direction both --hops 1

  # Context-only subgraph (no stratified sampling)
  python run_kgcore.py \
      --path_train /data/train_FG.txt \
      --path_test  /data/test.txt \
      --no_sampling

  # Early-exit diagnostic modes
  python run_kgcore.py --path_train ... --path_test ... --filter_mode literal_only
  python run_kgcore.py --path_train ... --path_test ... --filter_mode llm_only --use_LLM

Run `python run_kgcore.py --help` for the full flag reference.
Default hyperparameter values are documented in configs/default.yaml.
"""

from kg_core.args import parse_args
from kg_core.pipeline import run


if __name__ == "__main__":
    args = parse_args()
    run(args)
