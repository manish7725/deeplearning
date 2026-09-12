# Research Playbook

This repository is intended to end in research, not merely course completion.

## Paper-reading protocol

For every paper, write one page answering:

1. What problem is being solved?
2. What was believed before this work?
3. What is the new idea?
4. What assumptions does it make?
5. What is the mathematical object being optimized?
6. What is the baseline?
7. Which experiment actually supports the claim?
8. What alternative explanation exists?
9. What fails?
10. What experiment would falsify the central claim?

## Reproduction protocol

```text
paper
 ↓
exact dataset / preprocessing
 ↓
exact metric
 ↓
baseline
 ↓
implementation
 ↓
seed sweep
 ↓
comparison with reported result
 ↓
deviation analysis
```

## Ablation protocol

Change exactly one meaningful factor at a time unless the research question explicitly concerns an interaction.

Record:

- hypothesis;
- independent variable;
- dependent variable;
- controls;
- number of runs;
- compute budget;
- stopping rule;
- uncertainty;
- interpretation.

## Research notebook contract

A research notebook should make it possible for another learner to answer:

> What did you run, what did you change, what happened, and why should I believe the conclusion?

## PhD transition

A PhD-quality project is not defined by model size. It is defined by a precise question, credible evidence, reproducibility, and a result that changes what we know.
