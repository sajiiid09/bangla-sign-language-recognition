# Development Phases

## Phase 0: Environment and Tooling

Completed items documented across legacy status files:

- Python environment setup and dependency installation.
- CUDA verification and GPU-enabled training readiness.
- Training scripts for both primary and comparison tracks.

## Phase 1: Dataset Assembly and Processing

Documented outcomes:

- Combined manifest generation from multiple raw data sources.
- Landmark file generation and normalization workflows.
- Train/val/test split materialization.

Known caveat from legacy reports:

- At least part of prior training used placeholder/random landmark data in some workflows.

## Phase 2: Baseline System (BDSLW_SPOTER)

Documented work:

- Pose-based transformer pipeline with BdSL-specific normalization.
- Multi-phase notebook workflow for architecture, training, and evaluation.
- VRAM planning and deployment-focused profiling.

## Phase 3: Enhanced System (SignNet-V2)

Documented work:

- Multi-stream architecture design and implementation.
- Optimized training variant for difficult data conditions.
- Comparative framing against the SPOTER baseline.

## Phase 4: Experiment Tracking and Observability

Documented work:

- Weights and Biases integration in training and evaluation loops.
- Batch-level and epoch-level metric logging.
- Final-test metric and visualization logging.
- Setup and troubleshooting playbooks for authentication and runtime reliability.

## Phase 5: Data Quality Audits

Documented findings:

- Severe incompleteness in parts of the collected corpus.
- Per-class imbalance and subject/session gaps.
- Underrepresented expression categories.

This phase is critical because data issues are currently the largest bottleneck to reliable model quality.

## Phase 6: Current Priority Stack

1. Data integrity completion and rebalancing.
2. Reproducible aligned splits for both tracks.
3. Controlled baseline-vs-enhanced retraining.
4. Fair, artifact-backed comparison reporting.

## Phase 7: Production Readiness Gates

A run should only be considered production-ready when all are satisfied:

- Real extracted landmarks, not placeholders.
- Stable extraction toolchain with pinned versions.
- Cross-model comparison with identical split assumptions.
- Evaluation package includes confusion matrix and per-class metrics.
- Reproducible run metadata and checkpoint lineage.
