# Data Quality and Risks

## Audit Snapshot (Legacy Report)

A documented audit for the enhanced-model raw dataset reported severe incompleteness.

- Total files inspected: 775 videos
- Word classes: 55
- Subjects: 2
- Sessions: 2
- Repetitions: 5
- Expressions: 5
- Expected samples per word: 100
- Actual average per word: 14.09
- Estimated completion: about 14%

## Critical Findings

### 1. Dataset Incompleteness

- Roughly 86% of expected samples were missing in the audited scope.
- Some words had fewer than 10 samples, including extremely low-count classes.

### 2. Subject/Session Gaps

- One subject/session slice was reported as entirely missing for multiple words.
- Several words were reported with no samples from one subject.

### 3. Expression Imbalance

- Expression labels were unevenly distributed.
- Negation was significantly underrepresented versus neutral in the audit report.

## Modeling Impact

- High risk of overfitting for low-sample classes.
- Poor generalization for words with single-subject coverage.
- Split strategy constraints due to missing subject/session combinations.
- Unstable per-class metrics and confusing aggregate accuracy interpretation.

## Required Mitigations

1. Complete missing subject/session captures.
2. Increase samples for critically underrepresented classes.
3. Rebalance expression categories.
4. Apply class-weighted losses and robust augmentation while data is incomplete.
5. Use stratified split logic with explicit class-presence guarantees.

## Training-Time Safeguards

- Always publish class frequency table with results.
- Mark runs as exploratory when critical classes remain under-sampled.
- Use confidence intervals and per-class metrics, not only top-line accuracy.

## Readiness Criteria for Serious Benchmarking

A benchmark run should be treated as high-confidence only when:

- all classes exceed minimum sample thresholds,
- both subjects/sessions are represented according to protocol,
- expression labels are acceptably balanced,
- and split construction is reproducible and documented.
