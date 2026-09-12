# Deep Learning Syllabus

## Learning Principle
Build every concept from first principles:

**intuition → mathematics → matrix/tensor formulation → PyTorch implementation → experiment → practical example**

The series is intentionally written so a Class 7 student can follow the intuition while the mathematics remains correct.

---

# Part I — Foundations

- Python, NumPy and PyTorch
- Numbers, vectors and feature representations
- Matrices and matrix multiplication
- Linear transformations
- Tensors and tensor dimensions
- Probability, statistics and basic calculus

# Part II — Neural Networks

- Perceptron and neurons
- Forward propagation
- Activation functions
- Loss functions
- Matrix formulation of neural networks
- Computational graphs
- Backpropagation
- Automatic differentiation
- PyTorch autograd

# Part III — Optimization

- Gradient descent
- Learning rate and convergence
- SGD
- Momentum
- Adam
- Weight initialization
- Regularization
- Normalization
- Batch size
- Optimization scaling
- Training stability

# Part IV — Why Neural Networks Are Powerful

- Neural networks as function approximators
- Linear vs non-linear functions
- Width and depth
- Composition of functions
- Universal approximation theorem
- Expressive power of deep networks
- Geometry of deep learning
- Transformations
- Invariance
- Equivariance
- Inductive bias

# Part V — Convolutional Neural Networks

- Convolution and kernels
- Padding and stride
- Pooling
- CNN architectures
- Local structure
- Weight sharing
- Translation structure
- Transfer learning for vision

# Part VI — Graph Neural Networks

- Graphs, nodes and edges
- Adjacency matrices
- Node and edge features
- Message passing
- Aggregation
- Graph convolution
- Graph attention
- Permutation invariance
- Molecular, social and network examples

# Part VII — Sequence Models

- RNNs
- Hidden state
- Vanishing and exploding gradients
- LSTM
- GRU
- Sequence-to-sequence learning
- Memory in neural networks

# Part VIII — Attention and Transformers

- Embeddings
- Query, Key and Value
- Self-attention
- Multi-head attention
- Positional encoding
- Transformer encoder and decoder
- Language models
- Autoregressive prediction

# Part IX — Representation Learning

- What is a representation?
- Encoders and decoders
- Reconstruction-based learning
- Autoencoders
- Similarity learning
- Metric learning
- Contrastive learning
- Positive and negative pairs
- InfoNCE
- Hard negatives
- Self-supervised learning
- Representation geometry
- Architectural bias on representations

# Part X — Generalization and Robustness

- Training vs test performance
- Generalization
- Generalization gap
- Overfitting
- Bias and variance
- Overparameterization
- Double descent
- Inductive bias
- Distribution shift
- Out-of-distribution generalization
- Shortcut learning
- Adversarial robustness
- Data augmentation for robustness

# Part XI — Generative Deep Learning

- Generative modeling fundamentals
- Autoencoders
- Variational autoencoders
- GANs
- Diffusion models
- Latent variables
- Generative modeling + representation learning
- Conditional generative models
- Conditional GANs
- Conditional VAEs
- Conditional diffusion
- Image-to-image generation
- Text-to-image generation

# Part XII — Transfer Learning

- Pretrained models
- Feature extraction
- Frozen layers
- Linear probing
- Fine-tuning
- Knowledge distillation
- Foundation models
- Domain adaptation
- Transfer learning with data
- Prompting as adaptation

# Part XIII — Scaling Deep Learning

- Model size
- Dataset size
- Compute
- Batch-size scaling
- Learning-rate scaling
- Training steps
- Scaling laws
- Power-law behavior
- Compute-optimal training
- Data/model/compute tradeoffs

# Part XIV — Practical Deep Learning Engineering

- Dataset preparation
- Training and validation
- Evaluation metrics
- Experiment tracking
- GPU training
- Debugging data pipelines
- Tiny-dataset overfitting test
- Tensor-shape debugging
- Learning-rate experiments
- Training curves
- Model comparison
- Deployment considerations

# Part XV — Inference and Modern AI Systems

- Inference vs training
- Classification inference
- Autoregressive generation
- Greedy decoding
- Sampling
- Beam search
- In-context learning
- Test-time computation
- Search-based inference
- Measuring learned representations
- Evaluation beyond a single metric

# Part XVI — Learning From Human Preferences

- Prediction vs preference
- Reward signals
- Human preference data
- Reward models
- Policies
- Policy gradients
- PPO intuition
- Controlled policy updates
- Preference optimization for language models

# Part XVII — Advanced Theory

- Representation-learning theory
- Deep networks as function classes
- High-dimensional learning
- Kernel perspectives
- Metrized deep learning
- Measuring learned representations
- Research questions in modern deep learning

---

# Suggested Blog Progression

## Foundations

01. What Does It Mean for a Machine to Learn?
02. How Do Numbers Become Vectors?
03. Matrices — The Spreadsheet of Mathematics
04. A Matrix Can Transform Space
05. Meet the Smallest Neural Network
06. Why Does a Neuron Need an Activation Function?
07. Prediction Is Not the Same as Learning
08. Derivatives — A Compass for Learning
09. Gradient Descent — Teaching a Model to Improve
10. Backpropagation — Sending the Mistake Backward
11. Tensors — Numbers in Many Dimensions
12. How Do We Know If Our Model Really Learned?
13. How a Neural Network Learns to See — Convolution
14. When Order Matters — Learning From Sequences
15. How Can a Computer Represent the Meaning of a Word?
16. Attention — What Should I Look At?
17. Transformers — Building With Attention
18. How Does a Language Model Learn to Predict Text?
19. How Can a Machine Create Something New?
20. Build a Tiny Neural Network From Scratch

## Advanced Concepts Added From the MIT Deep Learning Comparison

21. How Does a Computer Calculate Gradients Automatically?
22. Can a Neural Network Learn Any Shape?
23. Why Should a Neural Network Care If We Move the Object?
24. When Data Looks Like a Network — Graph Neural Networks
25. Why Does a Neural Network Work on New Examples?
26. What Happens When We Make Training Bigger?
27. Can a Machine Discover a Good Way to Describe the World?
28. How Can a Machine Learn What Is Similar?
29. Can We Tell a Generative Model What to Create?
30. What Happens When the World Changes?
31. Can One Model Teach Another Model?
32. Why Do Bigger Models Often Learn More?
33. What Happens After a Model Is Trained?
34. The Detective's Guide to Training a Neural Network
35. Why Does the Shape of a Neural Network Matter?
36. How Do We Measure What a Neural Network Has Learned?
37. How Can We Teach a Model What Humans Prefer?

---

# Learning Principle

Every blog should move through the same loop:

```text
Story / intuition
      ↓
small example
      ↓
mathematics
      ↓
visual explanation
      ↓
matrix / tensor view
      ↓
PyTorch implementation
      ↓
experiment
      ↓
questions
      ↓
next concept
```

The goal is not to memorize formulas.

> **Understand what the formula is trying to say, then use the formula to make the idea precise.**