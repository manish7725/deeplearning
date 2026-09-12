# Deep Learning Curriculum — Class 8 → PhD

This curriculum is a staircase, not a list of topics. It starts with questions a Class 8 student can understand and progressively introduces the mathematics, implementation skills, systems knowledge and research habits needed for graduate-level deep learning.

## The learning loop

**Story → intuition → numerical example → mathematics → hand calculation → NumPy → PyTorch → visualization → controlled experiment → failure mode → project → paper → research question**

The level of mathematics and research rigor increases with every phase.

---

# Master Roadmap

| Phase | Level | Outcome |
|---|---|---|
| 0 | Class 8 | Think computationally and visualize mathematical ideas |
| 1 | Class 9–10 | Build mathematical foundations for ML |
| 2 | Class 11–12 | Become fluent in Python, linear algebra, probability and calculus |
| 3 | Undergraduate | Understand classical machine learning from first principles |
| 4 | First-principles DL | Derive and implement neural networks |
| 5 | Core architectures | Understand CNNs, sequence models, attention and graphs |
| 6 | Modern DL | Representation learning, generative models and foundation models |
| 7 | LLM engineering | Train, fine-tune, evaluate and serve modern language models |
| 8 | Systems | Understand GPU, distributed training, memory and inference systems |
| 9 | Graduate theory | Optimization, generalization, information theory, kernels and geometry |
| 10 | Research | Read, reproduce, ablate and critique papers |
| 11 | PhD | Formulate and execute original research |

---

# Phase 0 — Class 8: Learn to Think Like a Model

**Goal:** become comfortable turning real-world observations into numbers, rules and experiments.

1. What does it mean for a machine to learn?
2. Variables and observations
3. Numbers become vectors
4. Coordinates and graphs
5. Functions as machines
6. Matrices as tables of numbers
7. Matrix multiplication as repeated weighted addition
8. Linear transformations
9. Distance and similarity
10. Probability through games and uncertainty
11. Mean, median, spread and distributions
12. Python variables, loops and functions
13. NumPy arrays and shapes
14. Plotting a function
15. Build a tiny prediction machine

**Gate:** explain a simple predictor in words, calculate it by hand and reproduce it in Python.

---

# Phase 1 — Class 9–10: Mathematics for Learning

16. Algebra for models
17. Equations and rearranging formulas
18. Exponents and logarithms
19. Functions and composition
20. Coordinate geometry
21. Vectors and dot products
22. Matrix multiplication
23. Systems of equations
24. Geometry of projections
25. Slope and rate of change
26. Derivative intuition
27. Area under a curve
28. Basic probability
29. Conditional probability
30. Bayes' rule
31. Random variables
32. Mean, variance and covariance
33. Correlation versus causation
34. Sampling and estimation
35. Numerical error and floating point

**Gate:** solve small vector, probability and derivative problems without relying on a library.

---

# Phase 2 — Class 11–12 / Bridge to University

36. Multivariable functions
37. Partial derivatives
38. Gradients
39. Jacobians
40. Hessians
41. Chain rule
42. Vector spaces and bases
43. Linear independence and rank
44. Eigenvalues and eigenvectors
45. Singular Value Decomposition
46. Orthogonality and projections
47. Positive-definite matrices
48. Probability distributions
49. Expectation and conditional expectation
50. Maximum likelihood
51. Entropy and cross-entropy
52. KL divergence
53. Numerical optimization
54. Python scientific stack
55. NumPy vectorization
56. PyTorch tensors and autograd

**Gate:** derive a gradient, implement it twice, and verify it numerically with finite differences.

---

# Phase 3 — Undergraduate Machine Learning

57. Linear regression from first principles
58. Least squares geometry
59. Gradient descent for regression
60. Logistic regression
61. Maximum likelihood classification
62. Cross-entropy loss
63. Regularization and the bias-variance tradeoff
64. k-nearest neighbors
65. Decision trees
66. Ensembles and boosting
67. Support Vector Machines
68. Kernels
69. Principal Component Analysis
70. Gaussian mixture models
71. k-means and clustering
72. Naive Bayes
73. Generative versus discriminative modeling
74. Train/validation/test methodology
75. Metrics and calibration
76. Data leakage and experimental mistakes

**Gate:** implement linear regression, logistic regression and PCA from scratch and compare them with library implementations.

---

# Phase 4 — First-Principles Deep Learning

The repository's original 37 lessons form the spine of this phase.

77. What Does It Mean for a Machine to Learn?
78. How Do Numbers Become Vectors?
79. Matrices — The Spreadsheet of Mathematics
80. A Matrix Can Transform Space
81. Meet the Smallest Neural Network
82. Why Does a Neuron Need an Activation Function?
83. Prediction Is Not the Same as Learning
84. Derivatives — A Compass for Learning
85. Gradient Descent — Teaching a Model to Improve
86. Backpropagation — Sending the Mistake Backward
87. Tensors — Numbers in Many Dimensions
88. How Do We Know If Our Model Really Learned?
89. How a Neural Network Learns to See — Convolution
90. When Order Matters — Learning From Sequences
91. How Can a Computer Represent the Meaning of a Word?
92. Attention — What Should I Look At?
93. Transformers — Building With Attention
94. How Does a Language Model Learn to Predict Text?
95. How Can a Machine Create Something New?
96. Build a Tiny Neural Network From Scratch
97. How Does a Computer Calculate Gradients Automatically?
98. Can a Neural Network Learn Any Shape?
99. Why Should a Neural Network Care If We Move the Object?
100. When Data Looks Like a Network — Graph Neural Networks
101. Why Does a Neural Network Work on New Examples?
102. What Happens When We Make Training Bigger?
103. Can a Machine Discover a Good Way to Describe the World?
104. How Can a Machine Learn What Is Similar?
105. Can We Tell a Generative Model What to Create?
106. What Happens When the World Changes?
107. Can One Model Teach Another Model?
108. Why Do Bigger Models Often Learn More?
109. What Happens After a Model Is Trained?
110. The Detective's Guide to Training a Neural Network
111. Why Does the Shape of a Neural Network Matter?
112. How Do We Measure What a Neural Network Has Learned?
113. How Can We Teach a Model What Humans Prefer?

**Gate:** implement a multilayer network, derive backpropagation, train it, diagnose failures and explain every tensor shape.

---

# Phase 5 — Core Architecture Deep Dive

114. Convolution mathematically
115. Receptive fields
116. Padding, stride and dilation
117. Residual connections
118. Normalization layers
119. CNN design evolution
120. RNNs and recurrent computation graphs
121. Vanishing and exploding gradients
122. LSTM from equations
123. GRU from equations
124. Sequence-to-sequence learning
125. Teacher forcing
126. Encoder-decoder attention
127. Graph message passing
128. Graph convolution
129. Graph attention
130. Permutation invariance and equivariance
131. Neural operators
132. Set transformers and set functions

**Gate:** choose an architecture from data structure and inductive bias rather than fashion.

---

# Phase 6 — Representation and Generative Learning

133. Autoencoders
134. Variational inference intuition
135. Variational autoencoders
136. ELBO from first principles
137. GAN objectives
138. GAN instability
139. Diffusion as iterative denoising
140. Score matching
141. Noise schedules
142. Conditional generation
143. Latent diffusion
144. Image-to-image generation
145. Contrastive learning
146. InfoNCE
147. Hard negatives
148. Self-supervised learning
149. Masked modeling
150. Representation geometry
151. Disentanglement
152. Multimodal representation learning
153. CLIP-style learning
154. Vision transformers

**Gate:** implement at least one generative model and one self-supervised representation learner, then perform controlled ablations.

---

# Phase 7 — Transformers and LLMs

155. Tokenization
156. Embedding tables
157. Query, Key and Value
158. Scaled dot-product attention
159. Causal masking
160. Multi-head attention
161. Positional representations
162. Transformer blocks
163. Layer normalization and residual streams
164. Feed-forward networks
165. Logits and softmax
166. Autoregressive language modeling
167. Cross-entropy and perplexity
168. Pretraining objectives
169. Dataset construction and filtering
170. Data mixtures
171. Scaling laws
172. Compute-optimal training
173. In-context learning
174. Prompting as inference-time adaptation
175. Instruction tuning
176. Parameter-efficient fine-tuning
177. LoRA and adapters
178. Quantization
179. Mixture-of-Experts
180. Long-context methods
181. Retrieval-augmented generation
182. Tool use and agentic loops
183. LLM evaluation
184. Hallucination and factuality
185. Safety and robustness

**Gate:** build a small transformer from scratch, train it on a small corpus, inspect attention and loss curves, then fine-tune a pretrained model.

---

# Phase 8 — Deep Learning Systems

186. CPU versus GPU computation
187. GPU memory hierarchy
188. Tensor layouts and memory access
189. Vectorization and kernels
190. Mixed precision
191. Gradient accumulation
192. Activation checkpointing
193. Data parallelism
194. Distributed data parallelism
195. Model parallelism
196. Pipeline parallelism
197. Communication bottlenecks
198. Distributed optimization
199. Fault tolerance
200. Checkpoint design
201. Data pipelines
202. Reproducibility
203. Experiment tracking
204. Profiling training
205. Inference graphs
206. KV cache
207. Continuous batching
208. Quantized inference
209. Model serving
210. Latency-throughput-cost trade-offs

**Gate:** profile a training/inference workload and explain where its time and memory go.

---

# Phase 9 — Graduate Mathematical Theory

211. Optimization landscapes
212. Convex versus non-convex optimization
213. Smoothness and Lipschitz continuity
214. Strong convexity
215. SGD convergence intuition
216. Momentum dynamics
217. Adam and adaptive methods
218. Learning-rate schedules
219. Initialization theory
220. Normalization theory
221. Implicit regularization
222. Overparameterization
223. Double descent
224. PAC learning foundations
225. VC dimension
226. Stability and generalization
227. Algorithmic information perspectives
228. PAC-Bayes
229. Neural tangent kernels
230. Mean-field views of neural networks
231. Kernel versus feature learning
232. Random matrix theory intuition
233. High-dimensional geometry
234. Information bottleneck perspectives
235. Information theory of representations
236. Optimal transport
237. Wasserstein distances
238. Differential equations and neural ODEs
239. Dynamical systems views of deep learning
240. Symmetry groups
241. Invariance and equivariance
242. Geometric deep learning
243. RKHS and kernel methods
244. Functional analysis foundations

**Gate:** read a theoretical paper and reconstruct its assumptions, definitions, main derivation and proof strategy.

---

# Phase 10 — Research Engineering and Scientific Method

245. How to read a research paper
246. Finding the real claim
247. Reproducing a baseline
248. Dataset and preprocessing audits
249. Choosing metrics
250. Statistical significance
251. Confidence intervals
252. Multiple comparisons
253. Ablation design
254. Hyperparameter search without fooling yourself
255. Seed sensitivity
256. Compute budgets
257. Negative results
258. Reproducibility checklists
259. Benchmark contamination
260. Data provenance
261. Error analysis
262. Mechanistic probes
263. Representation probes
264. Scaling experiments
265. Writing a technical report
266. Making a research figure
267. Open-source research artifacts

**Gate:** reproduce a published result closely enough to explain every important discrepancy.

---

# Phase 11 — PhD Research Tracks

Choose one primary specialization after completing the common core.

## A. LLMs and Foundation Models
- Scaling and data laws
- Efficient transformers
- Long-context learning
- Retrieval and memory
- Multimodal models
- Agents and tool use
- Reasoning and test-time computation
- Synthetic data

## B. Generative Modeling
- Diffusion theory
- Flow matching
- Score-based methods
- Latent generative models
- Controllable generation
- Evaluation of generated distributions

## C. Representation Learning
- Self-supervision
- Contrastive objectives
- Disentanglement
- Geometry of learned spaces
- Multimodal representations
- Causal representation learning

## D. Optimization and Theory
- Implicit bias
- Generalization
- Optimization dynamics
- NTK
- Mean-field theory
- Sharpness and curvature
- Scaling theory

## E. Computer Vision
- Modern CNNs
- Vision transformers
- Detection
- Segmentation
- Video understanding
- 3D vision
- Vision-language models

## F. Graph and Geometric Learning
- Message passing limits
- Expressivity
- Graph transformers
- Equivariant networks
- Molecular learning
- Physical simulation

## G. Reinforcement Learning and Preference Learning
- Markov decision processes
- Dynamic programming
- Monte Carlo methods
- Temporal difference learning
- Policy gradients
- Actor-critic methods
- PPO
- Preference optimization
- Reward modeling
- Offline RL
- RLHF-style pipelines

## H. Interpretability and Safety
- Feature visualization
- Attribution
- Representation analysis
- Circuits
- Mechanistic interpretability
- Robustness
- Adversarial examples
- Alignment evaluation

## I. Efficient and Distributed AI
- Sparse computation
- MoE systems
- Quantization
- Distillation
- Distributed optimization
- Training/inference co-design

---

# PhD Dissertation Pipeline

A learner is research-ready when they can execute this loop independently:

```text
Observation
   ↓
Literature review
   ↓
Gap / contradiction
   ↓
Research question
   ↓
Falsifiable hypothesis
   ↓
Minimal experiment
   ↓
Baseline
   ↓
Ablation
   ↓
Statistical analysis
   ↓
Failure analysis
   ↓
Revision
   ↓
Reproduction by another person
   ↓
Paper / thesis
```

## Dissertation milestones

**Milestone 1:** reproduce 3 important papers.

**Milestone 2:** perform 3 meaningful ablations.

**Milestone 3:** identify a limitation in existing work.

**Milestone 4:** propose and test a modification.

**Milestone 5:** demonstrate improvement or a useful negative result.

**Milestone 6:** release code, data/configuration where possible, and reproducibility instructions.

**Milestone 7:** write the work as a research paper and defend the assumptions.

---

# Notebook Standard

Every executable lesson should contain, as appropriate:

1. Objective
2. Link to the corresponding blog
3. Prerequisites
4. Intuition
5. Hand calculation
6. NumPy implementation
7. PyTorch implementation
8. Tensor shapes
9. Visualization
10. Animation when it genuinely clarifies the concept
11. Sanity check
12. Controlled experiment
13. Failure mode
14. Exercises
15. Mini-project
16. Research extension

The learner should be able to run the notebook in Colab without needing hidden local setup.

# Blog Standard

Every blog should answer:

- Why was this idea needed?
- What problem does it solve?
- What is the simplest example?
- What does the mathematics mean?
- What assumptions are being made?
- How does it connect to the previous lesson?
- Where does it fail?
- What idea comes next?

The blog is the **textbook**, not a dump of notebook code.

# Mastery Rules

Do not advance because a lesson was read. Advance when the learner can:

- explain it without notes;
- calculate a small example by hand;
- implement the core operation;
- visualize what is happening;
- predict an experiment before running it;
- diagnose at least one failure mode;
- connect it to the next concept.

For graduate and PhD material, add:

- derive the central result;
- inspect assumptions;
- reproduce a result;
- design an ablation;
- critique the evidence;
- formulate a new question.

# The north star

> **Do not memorize deep learning. Reconstruct it.**

A strong learner should eventually be able to look at a new paper and ask:

**What problem is this solving? Why this objective? Why this architecture? What assumptions make it work? What would falsify the claim? How would I test it?**
