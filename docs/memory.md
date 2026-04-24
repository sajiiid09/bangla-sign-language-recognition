# Memory

## Repository Memory

This file stores durable project decisions, major incidents, and documentation traceability.

## Durable Decisions

1. Maintain a dual-track workflow: enhanced model as primary, SPOTER as comparison baseline.
2. Require skill selection before execution tasks whenever an applicable skill exists.
3. Enforce comparison claims only with artifact-backed metrics.
4. Treat data quality as a first-order constraint on performance conclusions.

## Major Incidents and Resolutions

### W&B Authentication Failure

- Symptom: missing authentication/init methods in `wandb` runtime.
- Root cause: corrupted package installation.
- Resolution: reinstall package and re-validate with setup/check scripts before training.

### Media/Feature Extraction Instability

- Symptom: extraction pipeline friction across MediaPipe versions.
- Impact: fallback to limited feature modes in some runs.
- Resolution path: pin toolchain for extraction and separate extraction validation from model tuning.

### Dataset Completeness Risk

- Symptom: severe sample incompleteness and imbalance in audited dataset slices.
- Impact: unstable and potentially misleading model metrics.
- Resolution path: prioritize collection completion and split integrity before final claims.

## Source-to-Target Traceability Map

This map records where each legacy markdown source was merged.

The full verbatim content of retired files is preserved in `docs/legacy-content-preservation.md`.

- `README.md` -> `docs/README.md`, `docs/development-phases.md`, `docs/training-operations.md`
- `PROJECT_STATUS.md` -> `docs/development-phases.md`, `docs/training-operations.md`, `docs/memory.md`
- `docs/BDSL_SPOTER_understanding.md` -> `docs/architecture.md`, `docs/model-comparison.md`
- `docs/BDSL_SPOTER_vram.md` -> `docs/architecture.md`
- `comparison model/BDSLW_SPOTER/README.md` -> `docs/architecture.md`, `docs/model-comparison.md`
- `new model/BdSL-Enhanced-SignNet/README.md` -> `docs/architecture.md`, `docs/model-comparison.md`
- `new model/BdSL-Enhanced-SignNet/TECHNICAL_REFERENCE.md` -> `docs/training-operations.md`, `docs/architecture.md`
- `new model/BdSL-Enhanced-SignNet/IMPROVEMENT_PLAN.md` -> `docs/development-phases.md`, `docs/model-comparison.md`
- `new model/BdSL-Enhanced-SignNet/DATASET_ISSUES_2026-01-31.md` -> `docs/data-quality-and-risks.md`
- `new model/BdSL-Enhanced-SignNet/READY_TO_USE.md` -> `docs/training-operations.md`
- `new model/BdSL-Enhanced-SignNet/ISSUE_RESOLVED.md` -> `docs/memory.md`, `docs/training-operations.md`
- `new model/BdSL-Enhanced-SignNet/WANDB_SETUP.md` -> `docs/training-operations.md`
- `new model/BdSL-Enhanced-SignNet/QUICKSTART_WANDB.md` -> `docs/training-operations.md`
- `new model/BdSL-Enhanced-SignNet/WANDB_INTEGRATION_SUMMARY.md` -> `docs/training-operations.md`
- `new model/BdSL-Enhanced-SignNet/WANDB_DOCS_INDEX.md` -> `docs/README.md`, `docs/training-operations.md`
- `new model/BdSL-Enhanced-SignNet/AUTHENTICATION_FIXED.md` -> `docs/memory.md`, `docs/training-operations.md`

## Ongoing Maintenance Rule

When new ad-hoc markdown files appear, merge durable content into canonical docs above and avoid reintroducing scattered standalone status files.
