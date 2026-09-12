# The Spine — What Each Chapter Inherits and Leaves

The course is one continuous argument, not 37 essays. This file is the contract that keeps
it that way:

> **The question a chapter leaves open in its final section must be the exact question the
> next chapter opens with.**

When writing chapter $N$, read row $N-1$ and row $N+1$ first. If your closing bridge does
not match the next chapter's inherited question, one of the two is wrong — fix it here
before writing prose.

Each chapter also carries a **running example** (`FORMAT.md` requires one). Where the example
continues from the previous chapter, the course gains coherence for free — the house that
was priced in Chapter 1 is the house that becomes a vector in Chapter 2.

## Part I — From arithmetic to a trained network (01–12)

| # | Chapter | Inherits the question | Teaches | Leaves open | Running example |
|---|---|---|---|---|---|
| 01 | What Does It Mean for a Machine to Learn? | — (the start) | model with dials, loss, landscape, slope, update rule | one number cannot describe a house | 4 house sales |
| 02 | Numbers Become Vectors | How do we hold many measurements as one object? | vector, dot product, norm, cosine, feature scaling | one house is a vector; how do we do *all* houses at once? | same houses, 3 features |
| 03 | Matrices: The Spreadsheet of Mathematics | How do we compute over a whole dataset at once? | matrix as stacked vectors, matrix–vector product, shapes | a matrix also *does* something to space | the house table |
| 04 | A Matrix Can Transform Space | What does a matrix *do*, not just store? | linear maps, rotation/scale/shear, composition | composition of linear maps is still linear — so depth buys nothing yet | a drawn square |
| 05 | Meet the Smallest Neural Network | What is the smallest unit that computes? | the neuron: weighted sum + bias | stacking neurons collapses back to one line | one neuron on the house data |
| 06 | Why Does a Neuron Need an Activation Function | Why does stacking gain nothing? | non-linearity, sigmoid/tanh/ReLU, why the collapse stops | which shapes can a network now represent? | XOR |
| 07 | *(see Sequencing Issues)* | | | | |
| 08 | Derivatives: The Compass for Learning | Chapter 1 used a slope without defining it | limits, $dy/dx$, rules, partial derivatives | slopes for one parameter — but a network has thousands | the §11 parabola from Ch 1 |
| 09 | Gradient Descent: Teaching a Model to Improve | How do we use slopes systematically? | gradient, step size, convergence, local minima | how do we get slopes *through* many layers? | loss surface |
| 10 | Backpropagation: Sending the Error Backward | How do we get every layer's slope? | chain rule, forward/backward pass, credit assignment | doing this by hand does not scale | 2-layer net by hand |
| 11 | Tensors: Numbers in Many Dimensions | Batches and layers need more than 2 axes | rank, shape, broadcasting, axis reasoning | we can compute — but is the model actually learning? | a batch of images |
| 12 | How Do We Know If Our Model Really Learned? | Loss went down. So what? | train/val/test, overfitting, baselines | fully-connected layers ignore structure in the data | memorizing vs generalizing |

## Part II — Structure in the data (13–19)

| # | Chapter | Inherits the question | Teaches | Leaves open | Running example |
|---|---|---|---|---|---|
| 13 | How a Neural Network Learns to See: Convolution | Why waste parameters on every pixel pair? | kernels, locality, weight sharing, receptive field | images have space; language has *order* | handwritten digits |
| 14 | When Order Matters: Learning From Sequences | How do we model order and memory? | recurrence, hidden state, vanishing gradients | words must first become numbers | a sentence |
| 15 | How Can a Computer Represent the Meaning of a Word? | What number *is* a word? | embeddings, similarity geometry (Ch 2's cosine, returning) | a fixed vector per word ignores context | word neighbourhoods |
| 16 | Attention: What Should I Look At? | How does context change meaning? | query/key/value, weighted lookup, softmax | attention alone is not an architecture | *"it"* in an ambiguous sentence |
| 17 | Transformers: Building With Attention | How do we build a full model from attention? | blocks, multi-head, residual, positional information | what objective do we train it on? | a tiny block |
| 18 | How Does a Language Model Learn to Predict Text? | What does it learn *from*? | next-token prediction, cross-entropy, perplexity | prediction gives generation — what else can we generate? | tiny corpus |
| 19 | How Can a Machine Create Something New? | Can a model produce rather than classify? | sampling, latent variables, autoencoders, the generative idea | we have used autograd all along without building it | generated digits |

## Part III — Building it yourself, then understanding why it works (20–26)

| # | Chapter | Inherits the question | Teaches | Leaves open | Running example |
|---|---|---|---|---|---|
| 20 | Build a Tiny Neural Network From Scratch | Can you write all of it, unaided? | full forward/backward/train loop, no framework | writing gradients by hand does not scale | MLP on toy data |
| 21 | Automatic Differentiation | How does a framework find gradients for you? | computation graph, reverse mode, a tiny autograd engine | what *can* these models represent, in principle? | scalar autograd |
| 22 | Neural Networks as Function Approximators | Which functions are reachable? | universal approximation, width vs depth, limits of the theorem | approximation says nothing about *generalization* | fitting curves |
| 23 | Geometry, Invariance, and Equivariance | Why does architecture matter at all? | symmetry, invariance, equivariance, inductive bias | some data is not a grid or a sequence | rotated images |
| 24 | Graph Neural Networks | What if the data is a network of relations? | message passing, permutation invariance, over-smoothing | why do over-parameterized models generalize at all? | a small graph |
| 25 | Why Neural Networks Generalize | Chapter 12 measured it; why does it happen? | capacity, implicit bias, double descent | if bigger works, how much bigger, and at what cost? | train/test curves |
| 26 | Scaling Rules for Training | How do we train big models stably? | batch size, LR schedules, normalization, optimizers | is there a law behind the gains? | LR sweeps |

## Part IV — Frontier practice (27–37)

| # | Chapter | Inherits the question | Teaches | Leaves open | Running example |
|---|---|---|---|---|---|
| 27 | Representation Learning | What makes a representation good? | features as learned geometry, probing | can we learn them without labels? | layer probes |
| 28 | Contrastive Learning | How do we learn without labels? | positives/negatives, InfoNCE, collapse | can we *steer* generation with a condition? | augmented pairs |
| 29 | Conditional Generative Models | How do we control what is generated? | conditioning, guidance, diffusion basics | models fail off-distribution — how badly? | conditioned samples |
| 30 | Out-of-Distribution and Robustness | What happens outside the training set? | shift, calibration, adversarial failure | must we retrain from scratch every time? | shifted test data |
| 31 | Transfer Learning and Fine-Tuning | How do we reuse what was learned? | pretraining, fine-tuning, LoRA/adapters | how big should the model and data be? | fine-tuned classifier |
| 32 | Scaling Laws | Is there a law relating compute, data and loss? | power laws, compute-optimal budgets | training is done — now it must *run* | loss vs compute |
| 33 | Inference Methods | How do we get good outputs efficiently? | decoding strategies, KV cache, latency | why do some models behave better than others of equal size? | sampling comparison |
| 34 | A Hacker's Guide to Deep Learning | Why did my training actually fail? | diagnostics, ablations, reproducibility | architecture choice shapes what is learned | a broken run |
| 35 | Architectural Bias and Representations | What does an architecture *prefer* to learn? | inductive bias measured empirically | can we measure representations rigorously? | architecture comparison |
| 36 | Metrized Deep Learning | How do we measure representation geometry? | metrics, distances between representations | models must match human preference, not just data | representation similarity |
| 37 | Preference Learning and Policy Optimization | How do we optimize for what people want? | reward models, DPO/RLHF, objective mismatch | *the course ends where research begins* | preference pairs |

---

## Sequencing issues to resolve

Three places where the current order violates the contract above. Each needs a decision
before the affected chapters are rewritten.

### 1. Chapter 07 is now redundant — **needs a decision**

`Lecture 07 - Prediction Is Not the Same as Learning` teaches a distinction that rewritten
Chapter 1 now makes in §5 and §18. Leaving it as-is means saying the same thing twice, six
chapters apart, which breaks the rule that every chapter answers a question the previous one
raised.

There is also a genuine **gap** in the current 37: squared error is the only loss taught
until cross-entropy appears mid-sentence in Chapter 18. Nothing explains why classification
needs a different loss at all.

**Recommendation:** repurpose slot 07 into **"Choosing What to Measure: Loss Functions"** —
why squared error fails for yes/no questions, probability as output, the discovery of
cross-entropy, and what a loss encodes about your intentions. It inherits naturally from
Chapter 6 (a neuron can now output a curve, so make it output a probability) and hands off
to Chapter 8 (this loss also needs a slope).

### 2. Chapters 20–22 sit after the application chapters

Building a network from scratch (20) and autodiff (21) are the natural consolidation of
backpropagation (10), but currently arrive after language models. As written in the table
above they still work — 20 re-enters after Chapter 19 as "now build all of it unaided" — but
a reader would be better served meeting them right after Chapter 11.

**Recommendation:** keep the numbering (renumbering churns every folder name, link and Colab
URL for a second time) and write 20–22 explicitly as a *consolidation arc*, as the table
above does. Revisit only if the reordering matters more than the churn.

### 3. Chapters 12 and 25 split one idea

Chapter 12 measures generalization; Chapter 25 explains it. That separation is defensible —
practice first, theory once there is enough machinery — provided Chapter 12 ends by
explicitly deferring the *why* to Chapter 25, and Chapter 25 opens by recalling Chapter 12's
measurement. Both bridges must say so out loud.
