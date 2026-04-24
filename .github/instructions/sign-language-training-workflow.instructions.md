---
description: "Use when training, fine-tuning, evaluating, or running data/model pipelines for Bangla sign-language recognition with emotion detection in this repository. Select and load the most relevant available skill(s) before execution."
name: "Sign Language Training Workflow"
---
# Sign Language + Emotion Model Workflow

- Treat this repository as a dual-track setup:
  - Primary track: new model/BdSL-Enhanced-SignNet
  - Comparison track: comparison model/BDSLW_SPOTER
- For every execution task (training, fine-tuning, evaluation, dataset processing, benchmark runs), you must identify relevant available skill(s) first and load them before taking action.
- If multiple skills are applicable, prefer the most specific skill for the requested task, then combine with supporting skills only when needed.
- When proposing or running experiments, include both:
  - How the enhanced model is handled
  - How results are compared against the comparison model
- Keep comparison runs reproducible and aligned with the same split/evaluation assumptions when possible.
- Modifying files in the comparison model is allowed when needed for fair benchmarking, compatibility, or requested experimentation.
- Before long-running training or tuning commands, state:
  - Which skill(s) were chosen
  - Why they match the task
  - Which model path(s) are being used
- If a requested task does not map to an available specialized skill, continue with standard coding/execution workflow and explicitly note that no matching specialized skill was available.
- Avoid making cross-model claims without a direct metric or run artifact.
- Prefer concise, execution-focused updates during long training workflows.
