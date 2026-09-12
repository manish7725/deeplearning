# The Spine — What Each Chapter Inherits and Leaves

The course is one continuous argument, not a pile of essays. This file is the contract that
keeps it that way:

> **The question a chapter leaves open in its final section must be the exact question the
> next chapter opens with.**

When writing chapter $N$, read row $N-1$ and row $N+1$ first. If your closing bridge does not
match the next chapter's inherited question, one of the two is wrong — fix it here before
writing prose.

**Status: 60 chapters.** The original spine was 37 and skipped the foundations that sit
*underneath* deep learning — probability, classical ML, evaluation, data handling. Those are
now first-class chapters rather than assumed knowledge. See [Migration](#migration-from-the-37-chapter-spine)
for how the old numbering maps onto the new.

---

## Part 0 — The Idea

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 01 | What Does It Mean for a Machine to Learn? | — | model with dials, loss, landscape, slope, update rule | one number cannot describe a house | 4 house sales |

## Part I — The Language: Linear Algebra

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 02 | Numbers Become Vectors | How do we hold many measurements as one object? | vector, **dot product**, norm, cosine, feature scaling | one house is a vector; how do we do *all* houses at once? | houses, 2 features |
| 03 | Matrices and Tensors | How do we compute over a whole dataset at once? | matrix, **matrix multiplication**, transpose, shape rule, scalar→vector→matrix→tensor, batch axis | a matrix also *does* something to space | the house table |
| 04 | A Matrix Can Transform Space | What does a matrix *do*, not just store? | linear maps, columns as basis images, basis/span, rotation/scale/shear/projection, **composition = multiplication**, determinant, rank, null space | some directions survive a transformation unchanged — which, and why? | a drawn square |
| 05 | Eigenvectors: What a Transformation Leaves Alone | Which directions does a matrix not rotate? | eigenvectors/values, diagonalization, SVD, PCA, low-rank structure | we can describe change in space; we cannot yet describe *rate* of change | a stretched grid |

## Part II — The Mathematics of Change

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 06 | Derivatives: The Compass for Learning | Chapter 1 used a slope without defining it | limits, $dy/dx$ from first principles, rules | a loss depends on *many* parameters, not one | the Ch 1 parabola |
| 07 | Partial Derivatives, Gradients and the Chain Rule | How does a slope work in many dimensions? | partial derivatives, gradient vector, directional derivative, **chain rule** | we know which way is downhill — how far do we step, and how often? | a 2-D loss surface |
| 08 | Gradient Descent: Teaching a Model to Improve | How do we use gradients systematically? | the algorithm, step size, convergence, local minima, saddle points | our loss was chosen by taste, not derived | loss surface |

## Part III — Uncertainty

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 09 | Describing Data: Mean, Variance, Distributions | Real prices scatter; what summarizes them? | mean, median, variance, std, covariance, correlation, the Gaussian | a summary is not a mechanism for *reasoning* about uncertainty | noisy house prices |
| 10 | Probability: Reasoning Under Uncertainty | How do we talk about what might happen? | events, conditional probability, independence, random variables, expectation | how should evidence *change* what we believe? | a noisy sensor |
| 11 | Bayes' Rule: What Evidence Does to Belief | How does data update belief? | Bayes, prior/likelihood/posterior, naive Bayes as a classifier | can belief-updating *derive* a loss function? | spam detection |
| 12 | Maximum Likelihood: Where Loss Functions Come From | Why squared error, and why not something else? | likelihood, MLE, **MSE from Gaussian noise**, **cross-entropy from Bernoulli**, MAP as regularization | we can now derive an objective — time to fit real models | Ch 1's loss, re-derived |

## Part IV — Classical Machine Learning

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 13 | Linear Regression, Properly | Chapter 1 fit a line by descent; is there an exact answer? | normal equations, closed form vs iterative, $R^2$, multicollinearity | a line cannot output a yes/no answer | house prices |
| 14 | Logistic Regression: The One-Neuron Network | How do we predict a *category*? | sigmoid, probability output, binary cross-entropy, decision boundary | this is literally one neuron — what about many classes? | will it sell? |
| 15 | Softmax and Cross-Entropy | How do we choose among many classes? | softmax, multi-class cross-entropy, logits, calibration | all of these draw straight boundaries | 3-way classification |
| 16 | Nearest Neighbours and the Curse of Dimensionality | Can we predict with no model at all? | kNN, distance metrics (Ch 2 returns), why high dimensions break intuition | memorizing all data does not scale or generalize | classifying houses |
| 17 | Decision Trees: Learning by Asking Questions | Can a model learn *rules* instead of weights? | splits, entropy, information gain, pruning | one tree overfits badly | a pricing tree |
| 18 | Ensembles: Forests and Boosting | How do we fix one unstable model? | bagging, random forests, boosting, gradient boosting/XGBoost | these dominate tabular data — so when do we need networks? | tree ensembles |
| 19 | Support Vector Machines and Kernels | What is the *best* boundary, not just a boundary? | margins, support vectors, the kernel trick, non-linear separation | kernels are hand-chosen; can features be *learned*? | separating two classes |

## Part V — Doing ML Honestly

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 20 | Train, Validation, Test | Loss went down. So what? | the three splits, overfitting, baselines, cross-validation | low loss is not the same as a *useful* model | memorize vs generalize |
| 21 | Evaluation Metrics: When 99% Accuracy Is Worthless | Which number actually measures success? | confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, class imbalance, regression metrics | metrics assume clean data — real data is not | rare-disease test |
| 22 | Data in the Real World | Where does the data itself go wrong? | encoding, missing values, outliers, scaling (Ch 2 returns), feature engineering, **leakage** | clean data still overfits | a messy dataset |
| 23 | Overfitting and Regularization | How do we stop memorizing? | L1/L2, weight decay, early stopping, data augmentation, the bias–variance tradeoff | linear models are still drawing straight lines | fitting noise |

## Part VI — Neural Networks

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 24 | Meet the Smallest Neural Network | What is the smallest unit that computes? | the neuron as $\mathbf{w}\cdot\mathbf{x}+b$, layers, forward pass | stacking layers collapses to one layer (Ch 4 said so) | one neuron |
| 25 | Why a Neuron Needs an Activation Function | Why does depth gain nothing? | non-linearity, sigmoid/tanh/ReLU/GELU, dead neurons | which functions can a network now represent? | XOR |
| 26 | Neural Networks as Function Approximators | What is reachable, in principle? | universal approximation, width vs depth, limits of the theorem | a theorem says it exists — training must still *find* it | fitting curves |
| 27 | Backpropagation: Sending the Error Backward | How does every layer learn its share? | chain rule through layers, forward/backward pass, credit assignment | doing this scalar-by-scalar does not scale | 2-layer net by hand |
| 28 | Matrix Backpropagation: $dW$, $db$, $dX$ by Hand | How do gradients work on whole matrices? | matrix calculus, $\partial L/\partial W$, $\partial L/\partial b$, $\partial L/\partial X$, shape checks as proof | writing gradients by hand does not scale either | dog/cat, 2 layers |
| 29 | Tensors in Practice | Batches and layers need more than 2 axes | broadcasting, reshaping, permuting, axis reductions, `einsum` | we can compute — can we build the whole thing unaided? | an image batch |
| 30 | Build a Neural Network From Scratch | Can you write all of it yourself? | full forward/backward/train loop, no framework | frameworks compute gradients automatically — how? | MLP on toy data |
| 31 | Automatic Differentiation | How does `loss.backward()` work? | computation graph, reverse mode, a tiny autograd engine | it trains — but badly, and slowly | scalar autograd |

## Part VII — Training Deep Networks

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 32 | Optimizers: SGD, Momentum, Adam | Plain descent is slow and unstable | mini-batches, SGD, momentum, AdaGrad, RMSProp, Adam, AdamW | optimizers cannot fix a badly scaled network | loss curves |
| 33 | Initialization and Normalization | Why do deep nets stall or explode? | Xavier/He init, batch norm, layer norm, residual connections | what else must scale with model size? | a 10-layer net |
| 34 | Learning Rates, Batch Size and Scaling Rules | How do the knobs interact? | schedules, warmup, gradient clipping, batch-size/LR relationships, mixed precision | training still fails for reasons the knobs cannot explain | LR sweeps |
| 35 | Training Diagnostics: A Hacker's Guide | Why did *my* run fail? | debugging recipes, ablations, seeds, reproducibility | fully-connected layers ignore the structure of the data | a broken run |

## Part VIII — Vision

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 36 | Convolution: How a Network Learns to See | Why waste parameters on every pixel pair? | kernels, stride/padding, weight sharing, pooling, receptive fields | shallow convnets stop improving | handwritten digits |
| 37 | CNN Architectures: LeNet to ResNet | How do we go deeper without breaking? | LeNet, VGG, residual connections, depth, feature hierarchies | why do these architectures work — what do they assume? | image classifier |
| 38 | Geometry, Invariance and Equivariance | What does an architecture *assume*? | symmetry, invariance, equivariance, inductive bias | images have space; language has order | rotated images |

## Part IX — Sequences and Language

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 39 | When Order Matters: Sequences and Memory | How do we model order? | recurrence, hidden state, BPTT, vanishing gradients | long-range memory fails | a sentence |
| 40 | LSTM and GRU: Fixing the Gradient | How do we remember longer? | gates, cell state, why gating preserves gradient | recurrence is sequential and cannot parallelize | long sequences |
| 41 | Embeddings: Representing Meaning | What number *is* a word? | embedding tables, word2vec, cosine geometry (Ch 2 returns) | one fixed vector per word ignores context | word neighbourhoods |
| 42 | Attention: What Should I Look At? | How does context change meaning? | query/key/value, scaled dot-product, softmax weights | attention alone is not an architecture | *"it"* in a sentence |
| 43 | Transformers: Building With Attention | How do we build a model from attention? | multi-head, blocks, residual + norm, positional encoding | what objective trains it? | a tiny block |
| 44 | Tokenization and How a Language Model Learns | What does it learn *from*? | tokenizers, next-token prediction, cross-entropy (Ch 15 returns), perplexity | a trained model must now *run* | tiny corpus |
| 45 | Inference: Decoding, Sampling and the KV Cache | How do we generate text well and cheaply? | greedy/beam/top-k/top-p, temperature, KV cache, latency | prediction gives generation — what else can we generate? | text generation |

## Part X — Representation and Generation

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 46 | Representation Learning | What makes a representation good? | features as learned geometry, probing, linear separability | can we learn them without labels? | layer probes |
| 47 | Contrastive Learning | How do we learn without labels? | positives/negatives, InfoNCE, SimCLR/CLIP, collapse | representations describe — can a model *create*? | augmented pairs |
| 48 | Autoencoders and the Generative Idea | Can a model produce, not just classify? | bottlenecks, autoencoders, VAEs, ELBO, sampling | generation is uncontrolled | generated digits |
| 49 | Conditional Generation and Diffusion | How do we control what is generated? | conditioning, guidance, GANs, diffusion forward/reverse | all of this assumed grid or sequence data | conditioned images |
| 50 | Graph Neural Networks | What if data is a network of relations? | message passing, permutation invariance, over-smoothing | why do over-parameterized models generalize at all? | a small graph |

## Part XI — Theory and Scale

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 51 | Why Neural Networks Generalize | Ch 20 measured it; why does it happen? | capacity, implicit bias, double descent, interpolation | if bigger is better, how much bigger? | train/test curves |
| 52 | Scaling Laws | Is there a law behind the gains? | power laws, compute-optimal budgets, data vs parameters | must we train from scratch every time? | loss vs compute |
| 53 | Transfer Learning and Fine-Tuning | How do we reuse what was learned? | pretraining, fine-tuning, LoRA/QLoRA, adapters | reused models meet data they never saw | fine-tuned classifier |
| 54 | Out-of-Distribution and Robustness | What happens outside the training set? | shift, calibration, adversarial examples, uncertainty | failures differ by architecture — why? | shifted test data |
| 55 | Architectural Bias and What Models Prefer to Learn | Why do equal-size models differ? | inductive bias measured empirically, shortcut learning | we need to *measure* representations, not describe them | architecture comparison |
| 56 | Measuring Representations | How do we compare what two models learned? | representation similarity, CKA, probing rigorously, metric geometry | models match data — but do they match *people*? | model comparison |

## Part XII — Frontier and Practice

| # | Chapter | Inherits | Teaches | Leaves open | Example |
|---|---|---|---|---|---|
| 57 | Preference Learning and Policy Optimization | How do we optimize for what people want? | reward models, RLHF, DPO, objective mismatch | a model alone is not a system | preference pairs |
| 58 | Retrieval, Tools and Agents | How does a model use knowledge it was not trained on? | embeddings + vector search, RAG, tool calling, agent loops | a system must run in production | a RAG pipeline |
| 59 | ML Systems: Serving, Quantization and Cost | How does this run at scale, affordably? | quantization, batching, throughput vs latency, monitoring, cost per token | everything so far was known; research is what is not | a served model |
| 60 | From Course to Research | You can build it — can you *extend* it? | reading papers, reproduction, ablation design, falsifiable questions | *the course ends where research begins* | a reproduction |

---

## Migration from the 37-chapter spine

Every original chapter keeps its content and moves to a new number. **23 chapters are new.**

| New | Source | New | Source | New | Source |
|---|---|---|---|---|---|
| 01–03 | old 01–03 ✅ *rewritten* | 21 | **new** | 41 | old 15 |
| 04 | old 04 | 22 | **new** | 42 | old 16 |
| 05 | **new** | 23 | **new** | 43 | old 17 |
| 06 | old 08 | 24 | old 05 | 44 | old 18 |
| 07 | **new** (split from old 08) | 25 | old 06 | 45 | old 33 |
| 08 | old 09 | 26 | old 22 | 46 | old 27 |
| 09 | **new** | 27 | old 10 | 47 | old 28 |
| 10 | **new** | 28 | **new** | 48 | old 19 |
| 11 | **new** | 29 | old 11 | 49 | old 29 |
| 12 | **new** | 30 | old 20 | 50 | old 24 |
| 13 | **new** | 31 | old 21 | 51 | old 25 |
| 14 | old 07 *repurposed* | 32 | **new** | 52 | old 32 |
| 15 | **new** | 33 | **new** | 53 | old 31 |
| 16 | **new** | 34 | old 26 | 54 | old 30 |
| 17 | **new** | 35 | old 34 | 55 | old 35 |
| 18 | **new** | 36 | old 13 | 56 | old 36 |
| 19 | **new** | 37 | **new** | 57 | old 37 |
| 20 | old 12 | 38 | old 23 | 58–60 | **new** |

**Old chapter 07** (*Prediction Is Not the Same as Learning*) is absorbed: rewritten Chapter 1
teaches that distinction in §5 and §18, so the slot becomes Logistic Regression — which is
where the course most needed a chapter, since $\sigma(\mathbf{w}\cdot\mathbf{x}+b)$ is exactly
one neuron and the true bridge from classical ML to networks.

## What the expansion fixes

The 37-chapter spine taught deep learning while assuming the layers underneath it. Concretely:

- **Cross-entropy** was used in old Ch 18 without ever being derived. Now Ch 12 derives it from
  maximum likelihood, and Ch 14–15 build it before any network needs it.
- **"Why squared error?"** — Chapter 1 answered honestly but incompletely ("it is a choice").
  Ch 12 completes it: MSE *is* maximum likelihood under Gaussian noise.
- **Classical ML** was absent. Logistic regression, trees, forests, boosting and SVMs now sit
  between statistics and networks, where they explain generalization more clearly than deep
  nets can.
- **Evaluation** was loss-only. Ch 21 covers the confusion matrix, precision/recall and ROC-AUC.
- **Data** was assumed clean. Ch 22 covers encoding, missing values, outliers and leakage.
- **Matrix backpropagation** was never done explicitly. Ch 28 derives $dW$, $db$ and $dX$ by hand.
- **Eigen/SVD** had no home. Ch 05 gives them one, immediately after transformations.

## The linear-algebra arc: Chapters 02–05

These chapters must leave the reader able to read every equation in the rest of the course.
Nothing later re-teaches them.

| | 02 Vectors | 03 Matrices & Tensors | 04 Transformations | 05 Eigen & SVD |
|---|---|---|---|---|
| **Core operation** | **dot product** — heart of a neuron | **matrix multiply** — heart of a layer | **composition** — heart of depth | **decomposition** — heart of structure |
| **Also** | norm, distance, cosine, orthogonality, feature scaling | matrix×vector, transpose, shape rule, tensors, rank as axis-count, batch axis | basis/span, rotation/scale/shear/projection, determinant, matrix rank, null space | eigenvectors, diagonalization, SVD, PCA, low-rank |
| **Ends asking** | how do we do every house at once? | what does a matrix *do* to space? | which directions survive unchanged? | how do we describe *rates* of change? |

Chapter 04's punchline decides the architecture of every network in the course: composing
linear maps yields another linear map, so a deep stack collapses into one matrix. That is
precisely the problem Chapters 24–25 exist to solve.

## Sequencing rules that must hold

1. **Chapter 12 before any loss is used.** MSE and cross-entropy must be derived before they
   are applied, or Chapter 1's honest "this is a choice" is never paid off.
2. **Chapter 20–23 before Part VI.** A reader must be able to tell a good model from a
   memorizing one *before* being handed a model with a million parameters.
3. **Chapters 12 and 51 split one idea.** Ch 20 measures generalization, Ch 51 explains it.
   Both bridges must say so out loud.
4. **Chapter 2's cosine returns in Ch 41**, and Chapter 2's feature scaling returns in Ch 22
   and Ch 33. Concepts recur by design — link back, never re-teach.
