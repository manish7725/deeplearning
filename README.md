# Deep Learning — Class 8 to PhD

A first-principles path from **school mathematics to deep-learning research**.

> **Blog = textbook. Notebook = laboratory. Experiments = discovery. Papers = research.**

## The learning loop

```text
Story → Intuition → Math → Hand calculation → NumPy → PyTorch
      → Visualization → Interactive experiment → Failure mode
      → Mini-project → Paper → Reproduction → Research question
```

Every important idea is taught at three levels:

1. **Class 8 intuition** — pictures, stories and tiny numbers.
2. **Engineering implementation** — NumPy/PyTorch and controlled experiments.
3. **Research mathematics** — derivations, assumptions, limitations, proofs and open questions.

## Curriculum

Start with [`SYLLABUS.md`](SYLLABUS.md). It is the master map from Class 8 through PhD-level research.

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

## Blog and laboratory

Each chapter is two files: `blog.md` carries the theory and mathematics, `notebook.ipynb`
carries the runnable code. Neither repeats the other — see [`FORMAT.md`](FORMAT.md) for the
layout both files follow and [`templates/`](templates/) for the starting points.

For each lesson:

1. Read the story and intuition.
2. Recalculate the smallest numerical example by hand.
3. Run the matching notebook.
4. Implement the idea with NumPy.
5. Implement it with PyTorch.
6. Visualize what the mathematics predicts.
7. Change one variable and observe the result.
8. Break the experiment intentionally and explain why.
9. Build the mini-project.
10. At advanced levels, reproduce a paper and formulate a research question.

## Existing first-principles spine

The repository already contains the 37-lesson mathematical spine from learning and vectors through transformers, generative models, representation learning, scaling, inference and preference learning. The new curriculum organizes that spine into a much larger progression instead of discarding it.

- 📚 **Chapters:** [`CHAPTERS.md`](CHAPTERS.md) — each numbered `Lecture NN - Title/` folder holds that chapter's `blog.md` and `notebook.ipynb` together
- 🧭 **Master syllabus:** [`SYLLABUS.md`](SYLLABUS.md)
- 📄 **Chapter format:** [`FORMAT.md`](FORMAT.md) — how `blog.md` and `notebook.ipynb` are laid out
- 🔗 **The spine:** [`SPINE.md`](SPINE.md) — what each chapter inherits from the last and leaves for the next
- 📐 **Learning contract:** [`LEARNING_CONTRACT.md`](LEARNING_CONTRACT.md)
- 🧪 **Interactive playground plan:** [`INTERACTIVE_PLAYGROUND.md`](INTERACTIVE_PLAYGROUND.md)
- 🧠 **Research methodology:** [`RESEARCH_PLAYBOOK.md`](RESEARCH_PLAYBOOK.md)

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
