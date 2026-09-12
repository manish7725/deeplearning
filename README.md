# Deep Learning — Class 8 to PhD

A first-principles path from **school mathematics to deep-learning research**.

> **Notebook = interactive textbook + laboratory. Markdown = parallel source. Applications = projects. Experiments = discovery. Papers = research.**

## The five-stage learning journey

This repository is intentionally **gated by difficulty and depth**. It is not a flat collection of 37 tutorials.

```text
STAGE 1  Foundations of Learning
   ↓ pass mastery gate
STAGE 2  Mathematics of Learning
   ↓ pass mastery gate
STAGE 3  Core Deep Learning
   ↓ pass mastery gate
STAGE 4  Deep Learning Theory & Modern Learning
   ↓ pass mastery gate
STAGE 5  Advanced Deep Learning & Research
```

👉 **[Open the five-stage curriculum](chapters/)**

👉 **[Read the stage goals and mastery gates](STAGES.md)**

### Stage map

| Stage | Focus | Chapters |
|---:|---|---:|
| 1 | Foundations of Learning | 01–07 |
| 2 | Mathematics of Learning | 08–12 |
| 3 | Core Deep Learning | 13–20 |
| 4 | Deep Learning Theory & Modern Learning | 21–30 |
| 5 | Advanced Deep Learning & Research | 31–37 |

**Rule:** finish a stage, pass its mastery gate, then continue. A learner should not be expected to understand Stage 5 material while still building Stage 1–2 foundations.

## Start learning

A Class 8 student should start from the **chapter notebook**, not from a separate blog page.

Every chapter is self-contained:

```text
Story → Intuition → Math → Hand calculation → NumPy → PyTorch
      → Visualization → Controlled experiment → Failure mode
      → Exercises → Application → Research bridge → Mastery check
```

The notebook is the complete lesson even when the student never opens the Markdown source.

## Chapter workspace

The learner-facing repository is organized by stage:

```text
chapters/
  stage-1-foundations-of-learning/
    README.md
    01-what-is-learning/
      01-what-is-learning.ipynb   ← START HERE
      01-what-is-learning.md      ← parallel source
      README.md
      apps/
    ...

  stage-2-mathematics-of-learning/
    08-derivatives-the-compass/
    ...

  stage-3-core-deep-learning/
    13-convolution/
    ...

  stage-4-deep-learning-theory-and-modern-learning/
    21-automatic-differentiation/
    ...

  stage-5-advanced-deep-learning-and-research/
    31-transfer-learning-and-fine-tuning/
    ...
```

Each chapter contains the notebook, parallel Markdown source, chapter README, and an `apps/` directory for applications. The notebook is the primary learning artifact.

## Master curriculum

The broader [`SYLLABUS.md`](SYLLABUS.md) remains the long-term Class 8 → PhD roadmap. The five-stage structure is the **gated learner-facing spine** for the current 37 first-principles lessons.

## Learning contract

A concept is not considered mastered merely because the learner read it. For each important idea, the learner should be able to:

1. explain it to a younger student;
2. calculate a small example by hand;
3. implement it in NumPy;
4. implement it in PyTorch when appropriate;
5. visualize or test its important behaviour;
6. explain at least one failure mode;
7. solve unfamiliar exercises;
8. connect it to the next abstraction.

At the advanced/research stage, add controlled ablations, reproducibility, uncertainty analysis, and falsifiable research questions.

## Source architecture

```text
blogs/NN-topic.md
       │
       │ canonical source
       ▼
lesson parser / curriculum builder
       │
       ├──────────────► notebooks/NN-topic.ipynb
       │                 legacy notebook collection
       │
       ▼
chapters/stage-N-.../NN-topic/
       ├── NN-topic.ipynb   ← learner-facing interactive textbook + lab
       ├── NN-topic.md      ← parallel source
       ├── README.md        ← chapter map + prerequisites
       └── apps/            ← applications built from the chapter
```

For now the original `blogs/` and `notebooks/` directories remain intact. They are migration/source assets; the staged `chapters/` tree is the learner-facing curriculum.

## Interactive mathematics

The target experience is an interactive mathematical playground wherever motion adds understanding:

- drag a point and see a derivative change;
- watch gradient descent move on a loss surface;
- change learning rate and observe convergence/divergence;
- inspect matrix multiplication inside a neural layer;
- visualize activations and decision boundaries;
- animate convolution windows;
- inspect attention weights token by token;
- visualize embeddings in 2D/3D;
- compare optimizers and initialization;
- fit scaling laws from measurements.

Interactive visuals are a second explanation, not a substitute for derivation.

## Research loop

```text
Question → hypothesis → baseline → measurement → intervention
→ controlled ablation → failure analysis → conclusion
→ reproducibility → paper → next question
```

## Branch policy

`main` is the stable published construct.

`reorg/5-stage-deep-learning` is the current experimental construct for this refactor. **When the old structure conflicts with the new five-stage design, the new design wins.**

Nothing is merged into `main` until the complete five-stage curriculum, all chapter notebooks, synchronization, links, applications, and mastery gates pass review.
