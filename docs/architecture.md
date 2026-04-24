# Architecture

## Repository Topology

This project is organized as a dual-track sign language research stack.

- Primary model track: `new model/BdSL-Enhanced-SignNet`
- Comparison model track: `comparison model/BDSLW_SPOTER`
- Shared data root: `Data/`

## End-to-End Data Flow

1. Raw video collection is organized by signer/source.
2. Landmark extraction generates per-sample arrays.
3. Normalization and split generation create training-ready sample lists.
4. Training/evaluation runs consume processed files and write checkpoints/artifacts.

## Primary Track: SignNet-V2 (Enhanced Model)

### Model Design

SignNet-V2 uses multi-stream spatiotemporal modeling and is designed to exceed a single-stream pose baseline.

- Streams:
  - Body pose: 99D
  - Hands: 126D
  - Face: 1404D
- Core blocks:
  - Stream-specific encoders
  - Cross-stream attention fusion
  - Hierarchical temporal encoding
  - Global transformer encoder
  - MLP classification head

### Practical Operating Modes

- Full multi-modal mode: body + hands + face.
- Pose-only fallback mode: body features only when full extraction is blocked by tooling/runtime constraints.

### Typical Configuration Envelope

- Transformer depth around 4-6 layers.
- Embedding sizes commonly 128-256.
- Regularization via dropout, mixup, label smoothing.
- Mixed precision training enabled when possible.

## Comparison Track: BdSLW_SPOTER Baseline

### Model Design

SPOTER is a pose-transformer baseline with BdSL-specific adaptations.

- Input features: 108D (54 landmarks x 2 coordinates).
- Typical architecture:
  - Positional encoding
  - Multi-head self-attention
  - Transformer encoder stack
  - Classification head for BdSL word classes

### Reported Capabilities in Existing Docs

- High top-1/top-5 results are documented for curated setups.
- Efficient runtime profile and consumer-GPU viability are documented.

## Hardware and VRAM Guidance

### Baseline VRAM Expectations (SPOTER docs)

- Batch 4: approximately 1.5-2 GB
- Batch 8: approximately 2.5-3 GB
- Batch 16: approximately 4-5 GB
- Batch 32: approximately 7-8 GB

### Primary Track Expectations (SignNet-V2 docs)

- Training: 8 GB+ VRAM recommended for full multi-stream settings.
- Inference: CPU-compatible; GPU improves latency and throughput.

## Architectural Risks and Constraints

- MediaPipe version drift impacts extraction workflow and reproducibility.
- Data incompleteness and class imbalance can dominate model behavior more than architecture changes.
- Reported metrics can differ strongly by dataset completeness, split strategy, and feature availability.

## Canonical Architecture Decision

Use both tracks as complementary systems:

- SignNet-V2 is the main research/production candidate.
- BDSLW_SPOTER remains the stable comparison baseline.

All experiments should retain cross-model comparability using aligned split assumptions and explicit metric reporting.
