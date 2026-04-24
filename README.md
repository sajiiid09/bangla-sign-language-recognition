# Bangla Sign Language Recognition

This repository contains a dual-track Bengali Sign Language (BdSL) research workflow:

- Primary track: enhanced model with emotion-aware recognition (`new model/BdSL-Enhanced-SignNet`)
- Comparison track: transformer baseline (`comparison model/BDSLW_SPOTER`)

All project documentation has been consolidated into the docs hub.

## Documentation Hub

Start here: [docs/README.md](docs/README.md)

Canonical docs:

- [docs/architecture.md](docs/architecture.md)
- [docs/development-phases.md](docs/development-phases.md)
- [docs/training-operations.md](docs/training-operations.md)
- [docs/model-comparison.md](docs/model-comparison.md)
- [docs/data-quality-and-risks.md](docs/data-quality-and-risks.md)
- [docs/agents.md](docs/agents.md)
- [docs/memory.md](docs/memory.md)
- [docs/legacy-content-preservation.md](docs/legacy-content-preservation.md)

## Quick Start

1. Install dependencies for the target model track.
2. Configure training environment and credentials.
3. Use the canonical training/evaluation process in [docs/training-operations.md](docs/training-operations.md).
4. Use fair benchmarking protocol in [docs/model-comparison.md](docs/model-comparison.md).

## Documentation Policy

- Avoid creating ad-hoc standalone markdown status files.
- Merge durable information into the canonical docs in `docs/`.
- Keep model comparison claims artifact-backed and split-aligned.
