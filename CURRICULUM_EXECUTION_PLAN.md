# Class 8 → PhD Curriculum Execution Plan

This branch is the **full experimental curriculum**. `main` is the stable baseline and is never modified by this work.

## Operating rule

> **New construct wins.** If an older lesson, notebook, folder, navigation pattern, or automation conflicts with this curriculum, replace the conflicting construct on this branch rather than preserving both versions.

The branch is developed continuously until every curriculum stage has a usable learning path.

## Lesson completion contract

A lesson is considered complete only when it has the appropriate artifacts:

1. **Theory blog** — motivation, Class-8 intuition, precise mathematics, hand calculation, visual explanation, previous/current/next bridge, misconceptions, failure modes, exercises and further study.
2. **Laboratory notebook** — matching blog, runnable Python, NumPy implementation where meaningful, PyTorch implementation where meaningful, assertions, visualizations, experiments, interpretation, challenges and mini-project.
3. **Interactive layer** — an embedded playground/animation when changing a mathematical variable or training state teaches more clearly than a static figure. Static derivation remains available.
4. **Research layer** — paper/reproduction/ablation material for advanced topics, with seeds, controls, metrics, compute/data notes and failure analysis where applicable.
5. **Navigation** — every lesson points backward and forward; notebook/blog links are branch-correct and unique.
6. **Quality** — deterministic validation passes and no stale branch/labs/transport markers remain.

## Build order

### Phase A — Foundation bridge

Class 8 → high-school mathematics → Python → first ML intuition.

Deliverables:
- beginner-friendly math explanations;
- small arithmetic examples;
- plotting and Python fundamentals;
- vector/matrix/probability/calculus preparation;
- first learning experiments.

### Phase B — Mathematical ML core

Linear algebra, probability/statistics, optimization and classical ML.

Deliverables:
- regression and classification from first principles;
- SVM, PCA, clustering and EM;
- numerical optimization experiments;
- library-vs-first-principles comparisons;
- a classical ML benchmark project.

### Phase C — First-principles deep learning

Upgrade the existing 37-lesson spine without breaking its conceptual sequence.

Priority order:
- 01–10: neuron → derivatives → gradient descent → backprop;
- 11–20: tensors → training → convolution → sequence → attention → generative models → from-scratch network;
- 21–29: autodiff → approximation → geometry → GNN → generalization → scaling → representation → contrastive → conditional generation;
- 30–37: robustness → transfer → scaling laws → inference → training diagnostics → architecture → metrics → preference learning.

Every lesson must be faithful to its own central mathematical example.

### Phase D — Core architectures

Optimization mechanics, CNNs, RNN/LSTM/GRU, residual networks, normalization, graph learning and vision transformers.

Deliverables:
- controlled architecture comparisons;
- gradient-flow experiments;
- receptive-field visualizations;
- sequence memory experiments;
- graph message-passing visualizations.

### Phase E — Representation and generative learning

Autoencoders, VAEs, GANs, diffusion, contrastive learning, masked modeling, self-distillation and multimodal representation learning.

Deliverables:
- objective derivations;
- toy implementations;
- latent-space visualizations;
- instability/failure demonstrations;
- reproduction-ready experiments.

### Phase F — LLMs and foundation models

Tokenization → embeddings → causal LM → transformer mathematics → pretraining → scaling → adaptation → modern architectures.

Deliverables:
- tiny tokenizer;
- tiny decoder-only transformer from tensor operations;
- attention/softmax/temperature playgrounds;
- KV-cache experiment;
- decoding comparison;
- LoRA/QLoRA experiment;
- RAG, MoE and multimodal system lessons.

### Phase G — Deep-learning systems

GPU mental model, kernels, memory, distributed training, sharding, mixed precision, quantization, inference and serving.

Deliverables:
- memory/compute accounting;
- profiling labs;
- distributed-training simulations where hardware is unavailable;
- quantization accuracy/latency experiments;
- serving and throughput benchmarks.

### Phase H — Graduate theory

Matrix calculus, information theory, statistical learning theory, optimization theory, kernels, NTK, mean-field limits, neural ODEs, optimal transport and representation geometry.

Deliverables:
- proofs or proof sketches at the appropriate level;
- numerical demonstrations of theoretical claims;
- counterexamples;
- symbolic/numerical gradient checks;
- bridges from undergraduate intuition to research mathematics.

### Phase I — Research methodology

Paper reading, contribution extraction, baseline reproduction, ablations, statistical analysis, benchmark hygiene, robustness and scientific writing.

Deliverables:
- reproducibility templates;
- experiment logs;
- research notebooks;
- negative-result documentation;
- workshop-style paper projects.

### Phase J — PhD research

Literature mapping → open problem → hypothesis → baseline → experiment → analysis → iteration → paper.

Deliverables:
- research-question templates;
- literature maps;
- falsification plans;
- multi-seed experiments;
- ablation matrices;
- final research portfolio.

## Iteration policy

Each scheduled run should:

1. inspect the current branch;
2. identify the highest-value incomplete dependency;
3. implement concrete repository changes;
4. update navigation and indexes;
5. run deterministic quality checks;
6. leave `main` untouched;
7. commit only to `reorg/class8-to-phd-curriculum`.

Large changes should be split into coherent commits. Do not mass-rewrite good educational prose merely to create activity; improve content where the learning contract identifies a real gap.

## Definition of done for the branch

The branch is ready for review when:

- all syllabus stages have explicit lesson mappings;
- every lesson has a theory/lab path appropriate to its level;
- the 37 first-principles lessons are internally coherent;
- mathematical playgrounds exist for high-value dynamic concepts;
- notebooks are reproducible and explain their code;
- advanced topics include genuine experiments rather than only descriptions;
- research lessons include reproduction and ablation practice;
- automated quality checks are green;
- no workflow can write to `main`;
- the repository can be followed as one continuous Class 8 → PhD learning journey.
