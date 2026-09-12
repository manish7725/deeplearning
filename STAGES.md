# Five-Stage Deep Learning Path

The repository is a **gated learning journey**, not a flat collection of tutorials.

A learner should complete a stage's mastery gate before progressing to the next stage. Later notebooks may reference earlier ideas, but they should not assume concepts that the learner has not yet passed.

## Stage 1 — Foundations of Learning

**Chapters 01–07**

1. What learning means
2. Numbers become vectors
3. Matrices
4. Linear transformations
5. The smallest neural network
6. Activation functions
7. Prediction vs learning

### Goal
Build the mental model of a machine-learning system: data → representation → model → prediction → error → learning.

### Gate
The learner can explain a small model in plain language, calculate a prediction by hand, trace tensor/vector shapes, and implement the basic computation in Python/NumPy.

---

## Stage 2 — Mathematics of Learning

**Chapters 08–12**

8. Derivatives
9. Gradient descent
10. Backpropagation
11. Tensors
12. Training and testing

### Goal
Understand how a neural network actually learns from an error signal.

### Gate
The learner can derive and calculate a small gradient by hand, perform gradient-descent updates, explain backpropagation, reason about tensor shapes, and distinguish training performance from generalization.

---

## Stage 3 — Core Deep Learning

**Chapters 13–20**

13. Convolution
14. Sequence models
15. Word embeddings
16. Attention
17. Transformers
18. Language models
19. Generative models
20. Neural network from scratch

### Goal
Understand the major computational patterns behind modern neural networks.

### Gate
The learner can implement a small neural network from first principles, explain convolution/sequence/attention computations numerically, trace a transformer block, and train a tiny model while diagnosing basic failures.

---

## Stage 4 — Deep Learning Theory & Modern Learning

**Chapters 21–30**

21. Automatic differentiation
22. Neural networks as function approximators
23. Geometry, invariance and equivariance
24. Graph neural networks
25. Why neural networks generalize
26. Scaling rules
27. Representation learning
28. Contrastive learning
29. Conditional generative models
30. Out-of-distribution and robustness

### Goal
Move from “how a model works” to “why the learning system behaves this way.”

### Gate
The learner can compare analytical derivatives with autodiff, reason about representation geometry, design controlled experiments, explain generalization and distribution shift, and reproduce a small representation-learning experiment.

---

## Stage 5 — Advanced Deep Learning & Research

**Chapters 31–37**

31. Transfer learning and fine-tuning
32. Scaling laws
33. Inference methods
34. Hacker's guide to deep learning
35. Architectural bias and representations
36. Measuring representations
37. Preference learning and policy optimization

### Goal
Develop the engineering judgment and research habits required to investigate modern deep-learning systems.

### Gate
The learner can formulate a hypothesis, establish a baseline, run controlled ablations, measure representations or inference behaviour, analyze failure modes, document reproducible experiments, and propose a research question.

---

# Progression rule

```text
Stage 1
  ↓ mastery gate
Stage 2
  ↓ mastery gate
Stage 3
  ↓ mastery gate
Stage 4
  ↓ mastery gate
Stage 5
```

**Do not optimize for finishing chapters. Optimize for passing the gate.**

Each stage should therefore contain:

- prerequisites;
- chapter sequence;
- self-checks;
- exercises;
- a stage project;
- a mastery assessment;
- an explicit gate before the next stage.

The notebooks remain the primary learning experience. Markdown remains the parallel source.
