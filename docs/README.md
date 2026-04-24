# Documentation Hub

This repository's documentation has been consolidated into a small structured set.

## Core Documents

- [architecture.md](architecture.md): System architecture for both models, data flow, and hardware guidance.
- [development-phases.md](development-phases.md): Project timeline, completed work, open gaps, and next phases.
- [training-operations.md](training-operations.md): End-to-end training operations, W&B setup, logging schema, and troubleshooting.
- [model-comparison.md](model-comparison.md): Baseline vs enhanced model comparison and fair benchmarking protocol.
- [data-quality-and-risks.md](data-quality-and-risks.md): Dataset quality audit, critical issues, and mitigation.
- [agents.md](agents.md): Agent execution rules for training/fine-tuning/evaluation tasks.
- [memory.md](memory.md): Decision history, resolved incidents, and source-to-target traceability map.
- [legacy-content-preservation.md](legacy-content-preservation.md): Verbatim archive of retired markdown files to ensure lossless transition.

## Consolidation Principles

- Information from legacy markdown files is preserved in categorized form.
- Duplicate and repetitive docs are merged into single canonical references.
- Operational guidance is centralized so model execution workflows remain reproducible.

## Scope

This hub covers both tracks in the repository:

- Primary track: `new model/BdSL-Enhanced-SignNet`
- Comparison track: `comparison model/BDSLW_SPOTER`

For command examples and run policy, start with [training-operations.md](training-operations.md) and [model-comparison.md](model-comparison.md).
