# Model Comparison

## Purpose

Maintain fair and reproducible comparison between:

- Enhanced model: `new model/BdSL-Enhanced-SignNet`
- Baseline model: `comparison model/BDSLW_SPOTER`

## Comparison Rules

1. Use aligned split assumptions whenever possible.
2. Report both aggregate and per-class metrics.
3. Include run artifacts and configuration snapshots.
4. Do not report improvement claims without direct metric evidence.

## Architecture Snapshot

### BDSLW_SPOTER Baseline

- Pose-centric transformer pipeline.
- Strong efficiency profile and established notebook workflow.
- Lower feature dimensionality relative to SignNet-V2.

### SignNet-V2 Enhanced

- Multi-stream body/hand/face representation.
- Cross-stream fusion and deeper temporal modeling.
- Larger model capacity with stronger augmentation options.

## Dataset and Split Considerations

The quality and completeness of dataset slices significantly affect conclusions.

Required report fields:

- sample counts per split
- signer/session coverage
- class frequency profile
- expression distribution profile

## Metrics to Report

### Primary

- Top-1 accuracy
- Macro F1
- Weighted F1
- Precision and Recall

### Secondary

- Top-5 accuracy
- Per-class accuracy
- Per-signer accuracy
- Confusion matrix

## Fair Run Protocol

1. Freeze dataset snapshot and split files.
2. Train baseline and enhanced with transparent configs.
3. Evaluate both with identical metric code where possible.
4. Publish side-by-side comparison table and artifacts.

## Known Interpretation Risks

- Placeholder features can invalidate comparison.
- Incomplete classes inflate metric volatility.
- Different split logic can create false gains.

## Decision Guidance

- If data quality is poor, treat results as exploratory.
- If splits are aligned and artifacts are complete, treat results as candidate evidence.
- Promote only results that are repeatable across reruns.
