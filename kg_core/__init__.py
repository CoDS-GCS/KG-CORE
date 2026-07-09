"""
KG-CORE: Knowledge Graph Context and Representative-Example Subgraph Sampler.

A direction-aware, LLM-guided pipeline for task-specific subgraph extraction,
targeting tail prediction (h, r, ?) in Graph Foundation Models.

Package layout
--------------
kg_core/
├── profiling/          Stage 1a — statistical analysis of the full KG
├── filtering/          Stage 1b — non-motif removal + LLM predicate selection
├── construction/       Stage 2  — sampling, expansion, and assembly
├── utils/              Shared I/O and performance logging
├── args.py             CLI argument parsing
└── pipeline.py         End-to-end orchestrator
"""

__version__ = "1.0.0"
