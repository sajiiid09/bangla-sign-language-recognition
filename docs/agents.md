# Agents

## Purpose

Define how coding/research agents should execute model and data tasks in this repository.

## Scope

Applies to:

- training
- fine-tuning
- evaluation
- data processing pipelines
- benchmark runs

for both tracks:

- `new model/BdSL-Enhanced-SignNet`
- `comparison model/BDSLW_SPOTER`

## Required Agent Workflow

1. Identify the task type.
2. Select and load the most relevant available skill(s) before execution.
3. State selected skill(s), why they fit, and which model path(s) will be used.
4. Execute with reproducible commands and explicit outputs.
5. Report metrics/artifacts with comparison context when relevant.

## Skill Selection Rules

- Prefer the most specific applicable skill first.
- Add supporting skills only when task complexity requires them.
- If no specialized skill exists, continue with standard workflow and explicitly state this.

## Comparison-Aware Execution

When running experiments or reporting outcomes:

- include both enhanced-model handling and comparison-model context,
- keep split/evaluation assumptions aligned whenever possible,
- avoid cross-model claims without direct metrics and run artifacts.

## Modification Policy

- Changes in comparison-model files are allowed when required for fair benchmarking, compatibility, or requested experiments.
- All such changes must be documented in run notes and comparison reports.

## Reporting Minimum

Every non-trivial execution report should include:

1. task objective
2. selected skill(s)
3. model path(s)
4. dataset/split context
5. key metrics and artifacts
6. limitations and known risks
