# Deep Learning — Class 8 to PhD

A first-principles learning path from the mathematics of a middle-school student to research-level deep learning.

> **Understand the idea → make the mathematics precise → calculate it by hand → implement it → visualize it → experiment → read the paper → ask a research question.**

## How to use this repository

| Layer | Purpose |
|---|---|
| 📖 **Blog** | The textbook: story, intuition, derivation, diagrams and the big picture |
| 🔬 **Notebook / Colab** | The laboratory: hand calculations, NumPy, PyTorch, plots, animations and experiments |
| 🧪 **Experiment** | Change one thing, measure the result, explain why it changed |
| 📄 **Paper** | Move from implementation to the research literature |
| 🎓 **Research** | Reproduce, ablate, challenge assumptions and formulate new questions |

**The blog explains. The notebook makes you calculate. The experiment makes you think.**

## Start here

1. **[Master syllabus](SYLLABUS.md)** — the complete Class 8 → PhD roadmap.
2. Start with **Blog 01** and follow the numbered sequence.
3. Open the matching notebook in Google Colab.
4. Do the small calculation before running the code.
5. Change parameters and predict what should happen before rerunning.
6. Keep an experiment log: hypothesis → setup → observation → explanation.

## Learning levels

### 🟢 Level 1 — Class 8 foundations
Numbers, variables, graphs, functions, vectors, matrices and beginner Python. The goal is curiosity and intuition, not advanced prerequisites.

### 🔵 Level 2 — High-school mathematics
Algebra, functions, geometry, probability, statistics, trigonometry and the calculus ideas needed to understand learning.

### 🟣 Level 3 — Undergraduate ML
Linear regression, logistic regression, loss functions, optimization, SVMs, PCA, probability and statistical learning.

### 🟠 Level 4 — First-principles deep learning
The existing 37-lesson spine: neurons → activations → loss → derivatives → gradient descent → backpropagation → tensors → CNNs → sequences → embeddings → attention → transformers → generative models → scaling → inference → preferences.

### 🔴 Level 5 — Graduate deep learning
Modern architectures, self-supervision, generative modeling, transfer learning, foundation models, efficient training and evaluation.

### ⚫ Level 6 — Research / PhD
Optimization theory, information theory, generalization, kernels, NTK, representation geometry, scaling theory, interpretability, alignment, distributed systems, paper reproduction and original research.

## The lesson contract

Every substantial lesson should aim for:

```text
Story
  ↓
Intuition
  ↓
Small numerical example
  ↓
Mathematical derivation
  ↓
Hand calculation
  ↓
NumPy implementation
  ↓
PyTorch implementation
  ↓
Visualization / animation
  ↓
Controlled experiment
  ↓
Failure mode
  ↓
Mini-project
  ↓
Paper connection
  ↓
Research question
```

Not every early lesson needs every research layer. The depth increases as the learner progresses.

## What “PhD level” means here

PhD depth is not achieved by adding difficult vocabulary. A research-ready learner should be able to:

- derive the central equations rather than memorize them;
- implement important algorithms from scratch;
- understand computational and statistical trade-offs;
- reproduce published results;
- design controlled ablations;
- distinguish correlation from causation in experiments;
- read papers critically;
- identify limitations and open problems;
- formulate a falsifiable hypothesis;
- build an experimental protocol that can answer it.

## Repository quality bar

New or rewritten material should be:

- mathematically correct;
- internally consistent with preceding lessons;
- runnable in a fresh Colab environment where code is promised;
- explicit about tensor shapes and units;
- accompanied by at least one sanity check;
- free of duplicated or cross-contaminated lesson content;
- linked to the next concept;
- clear about what is intuition versus theorem, approximation, empirical observation or research hypothesis.

## A practical study rhythm

For each lesson, use roughly:

**20 min read → 30 min calculate → 30 min code → 20 min experiment → 10 min explain it back.**

For advanced lessons, replace some reading time with paper reproduction and experiment design.

## Repository structure

```text
blogs/          # textbook-style lessons
notebooks/      # executable laboratories
SYLLABUS.md     # master curriculum and mastery gates
scripts/        # repeatable repository-quality automation
.github/        # automated quality checks
```

## Guiding principle

> **Do not learn deep learning as a list of architectures. Learn why each idea was invented, what mathematical problem it solves, what assumptions it makes, where it fails, and what came next.**
