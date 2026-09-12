# Deep Learning — Class 8 to PhD

A first-principles path from **school mathematics to deep-learning research**.

> **Notebook = interactive textbook + laboratory. Markdown = parallel source. Applications = projects. Experiments = discovery. Papers = research.**

## Start learning

A Class 8 student should start from the **chapter notebook**, not from a separate blog page.

👉 **[Open the 37 interactive chapters](chapters/)**

Each chapter is self-contained:

```text
Story → Intuition → Math → Hand calculation → NumPy → PyTorch
      → Visualization → Interactive experiment → Failure mode
      → Exercises → Application → Research question
```

The notebook is designed to work as the complete lesson even when the student never opens the Markdown source. This follows the computational-textbook model used by modern Jupyter-based educational resources: reusable narrative, executable mathematics, and interactive exploration. citeturn0search0

## Chapter workspace

Every topic now has one learner-facing folder:

```text
chapters/
  01-what-is-learning/
    README.md
    01-what-is-learning.ipynb   ← START HERE
    01-what-is-learning.md      ← parallel Markdown source
    apps/                        ← applications built from this chapter

  02-numbers-become-vectors/
    ...

  ...

  37-preference-learning-and-policy-optimization/
    ...
```

The chapter notebook is the primary learning artifact. The parallel Markdown copy remains intact for authoring, publishing, provenance, and future synchronization.

## Curriculum

Start with [`SYLLABUS.md`](SYLLABUS.md) for the master map from Class 8 through PhD-level research.

```text
LEVEL 0  Class 8 intuition + Python
LEVEL 1  High-school mathematics for ML
LEVEL 2  Undergraduate mathematics + classical ML
LEVEL 3  First-principles deep learning
LEVEL 4  Core architectures
LEVEL 5  Representation + generative learning
LEVEL 6  LLMs + foundation models
LEVEL 7  Deep-learning systems
LEVEL 8  Graduate mathematical theory
LEVEL 9  Research methodology
LEVEL 10 PhD research
```

## The learning loop

Every important idea is taught at three levels:

1. **Class 8 intuition** — pictures, stories and tiny numbers.
2. **Engineering implementation** — NumPy/PyTorch and controlled experiments.
3. **Research mathematics** — derivations, assumptions, limitations, proofs and open questions.

## Source architecture

```text
blogs/NN-topic.md
       │
       │ canonical source
       ▼
lesson parser / curriculum builder
       │
       ├──────────────► notebooks/NN-topic.ipynb
       │                 legacy/parallel notebook location
       │
       ▼
chapters/NN-topic/
       ├── NN-topic.ipynb   ← learner-facing interactive textbook + lab
       ├── NN-topic.md      ← parallel source
       ├── README.md        ← chapter map
       └── apps/            ← chapter applications
```

For now the original `blogs/` and `notebooks/` directories remain intact. The new `chapters/` tree is the learner-facing structure. Future cleanup can remove duplicate legacy copies after all links and workflows have migrated.

## Existing resources

- 📚 **[Interactive chapters](chapters/)** — primary learning path
- 📖 **[Source blogs](blogs/)** — parallel canonical Markdown source
- 📓 **[Legacy notebook collection](notebooks/)** — retained during migration
- 🧭 **[Master syllabus](SYLLABUS.md)**
- 📐 **[Learning contract](LEARNING_CONTRACT.md)**
- 🧪 **[Interactive playground plan](INTERACTIVE_PLAYGROUND.md)**
- 🧠 **[Research methodology](RESEARCH_PLAYBOOK.md)**

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

`reorg/class8-to-phd-curriculum` is the full experimental construct. **When the old structure conflicts with the new design, the new design wins.** This branch can be reshaped aggressively until the complete learning experience is ready.

Nothing is merged into `main` until the repository is reviewed as a complete curriculum.
