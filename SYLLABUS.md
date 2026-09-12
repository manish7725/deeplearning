# Class 8 → PhD Deep Learning Master Curriculum

This is the **master curriculum** for the repository. It is intentionally larger than the existing 37-blog spine. The 37 first-principles lessons remain the core mathematical backbone; this roadmap supplies the missing mathematics, classical machine learning, modern foundation-model topics, systems, theory and research practice around it.

## How to use this syllabus

Do not treat the numbers as a list to memorize. A module is mastered when you can:

- explain it to a younger student;
- calculate a small example by hand;
- implement it in NumPy;
- implement it in PyTorch when appropriate;
- visualize or test the important behaviour;
- explain at least one failure mode;
- solve unfamiliar exercises;
- connect it to the next abstraction.

At research level, add:

- reproduce a published result;
- perform a controlled ablation;
- quantify uncertainty;
- document compute/data/software versions;
- formulate a falsifiable research question.

---

# LEVEL 0 — CLASS 8: LEARN TO THINK LIKE A MODEL BUILDER

## 0.1 Numbers and patterns

1. Counting, comparison and magnitude
2. Positive and negative numbers
3. Fractions, decimals and percentages
4. Ratios and proportions
5. Powers and roots
6. Scientific notation
7. Units and dimensional reasoning
8. Sequences and patterns
9. Coordinates and the number line
10. Reading tables

## 0.2 Algebra without fear

11. Variables as boxes with names
12. Expressions and simplification
13. One-step equations
14. Multi-step equations
15. Linear relationships
16. Slope as rate of change
17. Intercepts
18. Systems of two equations
19. Inequalities
20. Functions as machines

## 0.3 Graphs and geometry

21. Plotting points
22. Reading a graph
23. Distance and direction
24. Area and volume
25. Transformations: shift, scale, rotate, reflect
26. Symmetry
27. Patterns in 2D
28. Visualizing data

## 0.4 Python as a calculator

29. Variables
30. Numbers and strings
31. Lists
32. Dictionaries
33. Conditions
34. Loops
35. Functions
36. Modules
37. Reading errors
38. Debugging with tiny examples
39. Files and CSV data
40. Basic plotting

## 0.5 First ML intuition

41. What is a feature?
42. What is a label?
43. Prediction vs rule
44. Training examples
45. Error as distance from the target
46. A model as a parameterized function
47. Why examples matter
48. Why memorization is not learning

**Mastery gate:** fit a straight line to five points using only arithmetic and Python, then explain why a line that perfectly memorizes training points can still fail on a new point.

---

# LEVEL 1 — HIGH-SCHOOL MATHEMATICS FOR MACHINE LEARNING

## 1.1 Algebra and functions

49. Polynomials
50. Factorization
51. Exponents
52. Logarithms
53. Exponential functions
54. Inverse functions
55. Composition of functions
56. Piecewise functions
57. Parametric equations
58. Coordinate geometry

## 1.2 Trigonometry

59. Angles
60. Sine, cosine and tangent
61. Unit circle
62. Periodic functions
63. Phase and frequency
64. Trigonometric identities

## 1.3 Vectors

65. Vector as an arrow
66. Vector as a list of numbers
67. Vector addition
68. Scalar multiplication
69. Dot product
70. Length and norm
71. Distance
72. Angle and cosine similarity
73. Projection
74. Orthogonality
75. Basis intuition

## 1.4 Matrices

76. Matrix as a table
77. Matrix addition
78. Matrix-vector multiplication
79. Matrix-matrix multiplication
80. Transpose
81. Identity matrix
82. Inverse intuition
83. Rank intuition
84. Linear systems
85. Geometry of matrix multiplication

## 1.5 Probability

86. Random experiments
87. Events
88. Probability rules
89. Conditional probability
90. Independence
91. Bayes' rule
92. Random variables
93. Expected value
94. Variance
95. Covariance
96. Correlation
97. Common distributions
98. Sampling

## 1.6 Calculus

99. Limits as zooming in
100. Derivative as local slope
101. Derivatives of basic functions
102. Product rule
103. Chain rule
104. Partial derivatives
105. Gradient
106. Directional derivative
107. Integral as accumulation
108. Multivariable surfaces
109. Taylor approximation

**Mastery gate:** derive the gradient of a two-variable quadratic, calculate it by hand at three points, and use it to predict the direction of steepest descent.

---

# LEVEL 2 — UNDERGRADUATE MATH + CLASSICAL MACHINE LEARNING

## 2.1 Linear algebra I

110. Vector spaces
111. Subspaces
112. Linear independence
113. Span
114. Basis and dimension
115. Coordinates in a basis
116. Linear maps
117. Kernel and image
118. Rank-nullity
119. Change of basis
120. Eigenvectors
121. Eigenvalues
122. Diagonalization
123. Symmetric matrices
124. Positive-definite matrices
125. Quadratic forms
126. Singular value decomposition
127. Low-rank approximation
128. Pseudoinverse

## 2.2 Probability and statistics

129. Joint distributions
130. Marginalization
131. Conditional distributions
132. Expectation of functions
133. Law of total expectation
134. Law of total variance
135. Maximum likelihood
136. Maximum a posteriori
137. Estimators
138. Bias and variance of estimators
139. Confidence intervals
140. Hypothesis testing
141. Bootstrap
142. Monte Carlo estimation
143. Maximum entropy intuition

## 2.3 Optimization

144. Objective functions
145. Convex sets
146. Convex functions
147. Gradient descent
148. Newton's method
149. Hessian
150. Constraints
151. Lagrange multipliers
152. KKT intuition
153. Stochastic optimization
154. Conditioning
155. Local vs global minima
156. Saddle points

## 2.4 Classical ML

157. Linear regression
158. Normal equations
159. Ridge regression
160. Lasso
161. Logistic regression
162. Softmax regression
163. Nearest neighbours
164. Decision trees
165. Random forests
166. Naive Bayes
167. Support vector machines
168. Kernels
169. Principal component analysis
170. Clustering
171. k-means
172. Gaussian mixtures
173. EM algorithm
174. Calibration
175. Feature engineering
176. Cross-validation
177. Data leakage
178. Baselines and metrics

**Mastery project:** build a classical ML benchmark containing linear regression, logistic regression, SVM and PCA from first principles, compare against library implementations, and explain every discrepancy.

---

# LEVEL 3 — FIRST-PRINCIPLES DEEP LEARNING

The existing 37 blogs form the central spine here.

179. What does it mean for a machine to learn?
180. Numbers become vectors
181. Matrices as computational objects
182. Linear transformations
183. The smallest neural network
184. Activation functions
185. Prediction vs learning
186. Derivatives as a learning compass
187. Gradient descent
188. Backpropagation
189. Tensors
190. Training and testing
191. Convolution
192. Sequence models
193. Word embeddings
194. Attention
195. Transformers
196. Language models
197. Generative models
198. Neural network from scratch
199. Automatic differentiation
200. Neural networks as function approximators
201. Geometry, invariance and equivariance
202. Graph neural networks
203. Generalization
204. Scaling rules
205. Representation learning
206. Contrastive learning
207. Conditional generation
208. Distribution shift and robustness
209. Transfer learning
210. Scaling laws
211. Inference methods
212. Training diagnostics
213. Architectural bias
214. Measuring representations
215. Preference learning

**Mastery gate:** implement a multilayer perceptron without a deep-learning framework, derive its gradients, train it on a tiny dataset, compare numerical gradients with analytical gradients, and explain each tensor shape.

---

# LEVEL 4 — CORE DEEP-LEARNING ARCHITECTURES

## 4.1 Optimization and training mechanics

216. SGD from sampling
217. Mini-batches
218. Momentum
219. Nesterov momentum
220. AdaGrad
221. RMSProp
222. Adam
223. AdamW
224. Learning-rate schedules
225. Warmup
226. Weight decay
227. Gradient clipping
228. Initialization
229. Xavier and He initialization
230. Batch normalization
231. Layer normalization
232. Residual connections
233. Gradient flow
234. Mixed precision
235. Reproducibility

## 4.2 CNNs and vision

236. Discrete convolution
237. Cross-correlation
238. Padding and stride
239. Receptive fields
240. Parameter sharing
241. Pooling
242. LeNet-style networks
243. Residual networks
244. Dense connections
245. Vision Transformers
246. Patch embeddings
247. Detection
248. Segmentation
249. Vision-language models

## 4.3 Sequences

250. Unrolled computation graphs
251. BPTT
252. Vanishing gradients
253. Exploding gradients
254. LSTM gates
255. GRU gates
256. Encoder-decoder sequence models
257. Teacher forcing
258. CTC intuition

## 4.4 Graph learning

259. Graph representations
260. Message passing
261. GCN
262. GraphSAGE
263. GAT
264. Over-smoothing
265. Graph positional information
266. Heterogeneous graphs
267. Graph-level prediction

---

# LEVEL 5 — REPRESENTATION, SELF-SUPERVISED AND GENERATIVE LEARNING

268. Autoencoders
269. Bottlenecks
270. Denoising autoencoders
271. Variational inference intuition
272. VAEs
273. ELBO
274. Reparameterization trick
275. GAN objective
276. Generator/discriminator dynamics
277. GAN instability
278. Diffusion forward process
279. Diffusion reverse process
280. Score functions
281. Noise schedules
282. Denoising objectives
283. Latent diffusion
284. Conditional generation
285. Image-to-image generation
286. Contrastive objectives
287. InfoNCE
288. Hard negatives
289. SimCLR-style learning
290. Masked modeling
291. Self-distillation
292. BYOL/DINO-style ideas
293. CLIP-style multimodal contrastive learning
294. Representation collapse
295. Probing representations
296. Linear probes
297. Nearest-neighbour geometry

---

# LEVEL 6 — LLMs AND FOUNDATION MODELS

## 6.1 Language representation

298. Tokenization
299. Subword vocabularies
300. Byte-level representations
301. Token IDs and embeddings
302. Positional information
303. Causal masking
304. Next-token prediction
305. Cross-entropy
306. Perplexity

## 6.2 Transformer mathematics

307. Query, key and value
308. Scaled dot-product attention
309. Softmax geometry
310. Multi-head attention
311. Feed-forward blocks
312. Residual streams
313. Layer normalization
314. Encoder-only models
315. Decoder-only models
316. Encoder-decoder models
317. KV cache
318. Context length
319. RoPE and relative position methods
320. Attention complexity
321. Memory complexity

## 6.3 Pretraining

322. Dataset construction
323. Deduplication
324. Data quality
325. Data mixtures
326. Curriculum over data
327. Distributed tokenization
328. Training objectives
329. Checkpointing
330. Evaluation during training
331. Scaling compute
332. Scaling data
333. Scaling model parameters
334. Chinchilla-style compute tradeoffs
335. Emergent behaviour: evidence and caution

## 6.4 Adaptation

336. Prompting
337. In-context learning
338. Supervised fine-tuning
339. Instruction tuning
340. Parameter-efficient fine-tuning
341. LoRA
342. QLoRA
343. Adapters
344. Preference optimization
345. Distillation
346. Model merging
347. Continued pretraining
348. Domain adaptation

## 6.5 Modern architectures

349. Mixture-of-experts
350. Routing
351. Sparse computation
352. Long-context architectures
353. Retrieval-augmented generation
354. Tool use
355. Multimodal models
356. Vision-language models
357. Audio-language models
358. Agents as systems

**Mastery project:** implement a tiny decoder-only transformer from tensor operations, train it on a small corpus, inspect attention, measure loss/perplexity, add KV caching, and compare several decoding strategies.

---

# LEVEL 7 — DEEP-LEARNING SYSTEMS

359. CPU vs GPU mental model
360. Parallelism
361. GPU memory hierarchy
362. Tensor layouts
363. Matrix multiplication kernels
364. CUDA execution model
365. Streams and synchronization
366. PyTorch eager execution
367. Graph compilation
368. Kernel fusion
369. Data loading pipelines
370. Pinned memory
371. Distributed data parallelism
372. All-reduce
373. Data parallelism
374. Tensor parallelism
375. Pipeline parallelism
376. Expert parallelism
377. ZeRO-style sharding
378. Checkpoint sharding
379. Activation checkpointing
380. Gradient accumulation
381. Mixed precision and numerical stability
382. Quantization
383. INT8/INT4 concepts
384. Weight-only quantization
385. KV-cache memory
386. Inference batching
387. Continuous batching
388. Speculative decoding
389. Model serving
390. Latency vs throughput
391. Profiling
392. Cost per token
393. Fault tolerance
394. Experiment tracking
395. Dataset/version management

---

# LEVEL 8 — GRADUATE MATHEMATICAL THEORY

## 8.1 Matrix calculus

396. Jacobians
397. Vector-Jacobian products
398. Jacobian-vector products
399. Hessians
400. Matrix derivatives
401. Trace tricks
402. Differential notation
403. Reverse-mode autodiff
404. Forward-mode autodiff
405. Higher-order differentiation

## 8.2 Information theory

406. Entropy
407. Cross-entropy
408. KL divergence
409. Jensen-Shannon divergence
410. Mutual information
411. Conditional mutual information
412. Data processing inequality
413. Rate-distortion intuition
414. Information bottleneck

## 8.3 Statistical learning theory

415. Empirical risk minimization
416. Uniform convergence
417. VC dimension
418. Rademacher complexity
419. PAC learning
420. Stability
421. Algorithmic stability
422. Margin theory
423. Compression views
424. Double descent
425. Interpolation

## 8.4 Optimization theory

426. Smoothness
427. Lipschitz continuity
428. Strong convexity
429. Convergence rates
430. Stochastic approximation
431. Noise in SGD
432. SDE views of optimization
433. Sharpness
434. Flatness controversies
435. Loss landscapes
436. Saddle-point dynamics

## 8.5 Modern theory

437. Kernel methods
438. RKHS
439. Neural tangent kernels
440. Lazy training
441. Mean-field limits
442. Infinite-width networks
443. Feature learning vs kernel regimes
444. Neural ODEs
445. Dynamical systems views
446. Optimal transport
447. Wasserstein distance
448. Geometry of representations
449. Manifold hypotheses
450. Symmetry and group actions
451. Equivariance theory
452. Category-inspired perspectives

---

# LEVEL 9 — RESEARCH METHODOLOGY

453. How to read a paper
454. Finding the actual contribution
455. Separating evidence from claims
456. Reproducing a baseline
457. Experimental controls
458. Ablation design
459. Random seeds
460. Confidence intervals
461. Statistical significance vs practical significance
462. Effect sizes
463. Multiple comparisons
464. Benchmark contamination
465. Data leakage in research
466. Evaluation design
467. Adversarial evaluation
468. Stress testing
469. Robustness evaluation
470. Scaling experiments
471. Compute accounting
472. Carbon/cost accounting
473. Reproducible environments
474. Dataset cards
475. Model cards
476. Experiment logs
477. Negative results
478. Research notebooks
479. Scientific writing
480. Figures that explain mechanisms
481. Peer-review simulation
482. Research ethics

## Research project ladder

483. Reproduce a simple published experiment
484. Reproduce a medium deep-learning paper
485. Reproduce a transformer result
486. Reproduce a scaling-law result
487. Perform a three-variable ablation
488. Identify a failure mode
489. Build a hypothesis around the failure
490. Design a falsification test
491. Run multiple seeds
492. Write a workshop-style paper

---

# LEVEL 10 — PHD RESEARCH

493. Choose a research area
494. Build a literature map
495. Identify the unresolved assumption
496. Formulate a precise research question
497. Define measurable hypotheses
498. Establish baselines
499. Build an experimental harness
500. Design ablations before running them
501. Quantify uncertainty
502. Investigate negative results
503. Distinguish correlation from mechanism
504. Develop theory where useful
505. Develop new algorithms where useful
506. Test generalization across datasets
507. Test scaling behaviour
508. Stress-test conclusions
509. Reproduce your own result
510. Release code/data when possible
511. Write a technical report
512. Write a conference-style paper
513. Present and defend the work
514. Build a thesis narrative
515. Identify the next research problem

---

# CAPSTONE PROJECTS

## Capstone A — Learn a line

Build linear regression from arithmetic → NumPy → PyTorch. Animate gradient descent.

## Capstone B — Neural network from scratch

Build forward propagation, loss, backpropagation and SGD without autograd.

## Capstone C — Vision laboratory

Build a CNN, visualize filters/receptive fields, and compare it with a simple MLP.

## Capstone D — Tiny language model

Train a small transformer, inspect attention and implement decoding.

## Capstone E — Representation laboratory

Train an encoder with contrastive learning and measure embedding geometry.

## Capstone F — Generative laboratory

Implement a tiny VAE and diffusion model and compare their objectives and failure modes.

## Capstone G — Systems laboratory

Profile a model, measure memory/latency/throughput, quantize it, and serve it.

## Capstone H — Research reproduction

Choose a paper, reproduce one central result, document deviations, run an ablation and write a reproduction report.

## Capstone I — Original research

Propose, test and defend one falsifiable hypothesis. The goal is not to make a large model; the goal is to discover something reliable.

---

# THE COMPLETE LEARNING CONTRACT

For a normal lesson:

```text
story
 ↓
intuition
 ↓
tiny numerical example
 ↓
mathematical definition
 ↓
hand derivation
 ↓
matrix/tensor form
 ↓
NumPy implementation
 ↓
PyTorch implementation
 ↓
visualization / animation
 ↓
controlled experiment
 ↓
failure mode
 ↓
exercise
 ↓
mini-project
 ↓
next lesson
```

For an advanced lesson:

```text
concept → derivation → implementation → benchmark
→ paper → reproduction → ablation → uncertainty
→ failure analysis → research question
```

**The standard is understanding, not completion percentage.**
