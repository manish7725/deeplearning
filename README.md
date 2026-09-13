# Deep Learning — From First Principles

A course that starts at Class-7 intuition and ends at research-level understanding.
Sixty chapters, written as one continuous argument: **every chapter answers the question
the previous one left open.**

> Blog = the lecture. Notebook = the laboratory. Nothing is asserted that is not derived,
> and nothing is claimed in the lecture that the notebook does not check.

## How to use a chapter

Each chapter is one folder holding `blog.md` and `notebook.ipynb`.

1. Read `blog.md` — the story, the mathematics, the derivations.
2. Recalculate its smallest numerical example by hand.
3. Open `notebook.ipynb` in Colab and **predict each result before running it**.
4. Change one variable, break it deliberately, and explain what happened.

Chapters are authored with the [`write-chapter`](.claude/skills/write-chapter/SKILL.md)
skill, which carries the writing standard, the full spine and the verification rules.

| | Status |
|---|---|
| ✅ | written to the current standard |
| 📄 | drafted — awaiting rewrite to the standard |
| ○ | planned — not yet written |

## Chapters

### Part 0 — The Idea

01 ✅ [What Does It Mean for a Machine to Learn](<Lecture 01 - What Does It Mean for a Machine to Learn/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2001%20-%20What%20Does%20It%20Mean%20for%20a%20Machine%20to%20Learn/notebook.ipynb)  

### Part I — The Language: Linear Algebra

02 ✅ [Numbers Become Vectors](<Lecture 02 - Numbers Become Vectors/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2002%20-%20Numbers%20Become%20Vectors/notebook.ipynb)  
03 ✅ [Matrices: The Spreadsheet of Mathematics](<Lecture 03 - Matrices: The Spreadsheet of Mathematics/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2003%20-%20Matrices%3A%20The%20Spreadsheet%20of%20Mathematics/notebook.ipynb)  
04 ✅ [A Matrix Can Transform Space](<Lecture 04 - A Matrix Can Transform Space/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2004%20-%20A%20Matrix%20Can%20Transform%20Space/notebook.ipynb)  
05 ✅ [Eigenvectors: What a Transformation Leaves Alone](<Lecture 05 - Eigenvectors: What a Transformation Leaves Alone/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2005%20-%20Eigenvectors%3A%20What%20a%20Transformation%20Leaves%20Alone/notebook.ipynb)  

### Part II — The Mathematics of Change

06 📄 [Derivatives: The Compass for Learning](<Lecture 06 - Derivatives: The Compass for Learning/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2006%20-%20Derivatives%3A%20The%20Compass%20for%20Learning/notebook.ipynb)  
07 📄 [Partial Derivatives, Gradients and the Chain Rule](<Lecture 07 - Partial Derivatives, Gradients and the Chain Rule/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2007%20-%20Partial%20Derivatives%2C%20Gradients%20and%20the%20Chain%20Rule/notebook.ipynb)  
08 📄 [Gradient Descent: Teaching a Model to Improve](<Lecture 08 - Gradient Descent: Teaching a Model to Improve/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2008%20-%20Gradient%20Descent%3A%20Teaching%20a%20Model%20to%20Improve/notebook.ipynb)  

### Part III — Uncertainty

09 ○ Describing Data: Mean, Variance, Distributions  
10 ○ Probability: Reasoning Under Uncertainty  
11 ○ Bayes' Rule: What Evidence Does to Belief  
12 📄 [Maximum Likelihood: Where Loss Functions Come From](<Lecture 12 - Maximum Likelihood: Where Loss Functions Come From/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2012%20-%20Maximum%20Likelihood%3A%20Where%20Loss%20Functions%20Come%20From/notebook.ipynb)  

### Part IV — Classical Machine Learning

13 📄 [Linear Regression, Properly](<Lecture 13 - Linear Regression, Properly/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2013%20-%20Linear%20Regression%2C%20Properly/notebook.ipynb)  
14 📄 [Logistic Regression: The One-Neuron Network](<Lecture 14 - Logistic Regression: The One-Neuron Network/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2014%20-%20Logistic%20Regression%3A%20The%20One-Neuron%20Network/notebook.ipynb)  
15 📄 [Softmax and Cross-Entropy](<Lecture 15 - Softmax and Cross-Entropy/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2015%20-%20Softmax%20and%20Cross-Entropy/notebook.ipynb)  
16 📄 [Nearest Neighbours and the Curse of Dimensionality](<Lecture 16 - Nearest Neighbours and the Curse of Dimensionality/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2016%20-%20Nearest%20Neighbours%20and%20the%20Curse%20of%20Dimensionality/notebook.ipynb)  
17 📄 [Decision Trees](<Lecture 17 - Decision Trees/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2017%20-%20Decision%20Trees/notebook.ipynb)  
18 📄 [Ensembles: Forests and Boosting](<Lecture 18 - Ensembles: Forests and Boosting/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2018%20-%20Ensembles%3A%20Forests%20and%20Boosting/notebook.ipynb)  
19 📄 [Support Vector Machines and Kernels](<Lecture 19 - Support Vector Machines and Kernels/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2019%20-%20Support%20Vector%20Machines%20and%20Kernels/notebook.ipynb)  

### Part V — Doing ML Honestly

20 📄 [How Do We Know If Our Model Really Learned](<Lecture 20 - How Do We Know If Our Model Really Learned/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2020%20-%20How%20Do%20We%20Know%20If%20Our%20Model%20Really%20Learned/notebook.ipynb)  
21 ○ Evaluation Metrics: When 99% Accuracy Is Worthless  
22 ○ Data in the Real World: Encoding, Missing Values, Leakage  
23 ○ Overfitting and Regularization  

### Part VI — Neural Networks

24 📄 [Meet the Smallest Neural Network](<Lecture 24 - Meet the Smallest Neural Network/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2024%20-%20Meet%20the%20Smallest%20Neural%20Network/notebook.ipynb)  
25 📄 [Why Does a Neuron Need an Activation Function](<Lecture 25 - Why Does a Neuron Need an Activation Function/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2025%20-%20Why%20Does%20a%20Neuron%20Need%20an%20Activation%20Function/notebook.ipynb)  
26 📄 [Neural Networks as Function Approximators](<Lecture 26 - Neural Networks as Function Approximators/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2026%20-%20Neural%20Networks%20as%20Function%20Approximators/notebook.ipynb)  
27 📄 [Backpropagation: Sending the Error Backward](<Lecture 27 - Backpropagation: Sending the Error Backward/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2027%20-%20Backpropagation%3A%20Sending%20the%20Error%20Backward/notebook.ipynb)  
28 📄 [Matrix Backpropagation: dW, db, dX by Hand](<Lecture 28 - Matrix Backpropagation: dW, db, dX by Hand/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2028%20-%20Matrix%20Backpropagation%3A%20dW%2C%20db%2C%20dX%20by%20Hand/notebook.ipynb)  
29 📄 [Tensors: Numbers in Many Dimensions](<Lecture 29 - Tensors: Numbers in Many Dimensions/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2029%20-%20Tensors%3A%20Numbers%20in%20Many%20Dimensions/notebook.ipynb)  
30 📄 [Build a Tiny Neural Network From Scratch](<Lecture 30 - Build a Tiny Neural Network From Scratch/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2030%20-%20Build%20a%20Tiny%20Neural%20Network%20From%20Scratch/notebook.ipynb)  
31 📄 [Automatic Differentiation](<Lecture 31 - Automatic Differentiation/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2031%20-%20Automatic%20Differentiation/notebook.ipynb)  

### Part VII — Training Deep Networks

32 ○ Optimizers: SGD, Momentum, Adam  
33 ○ Initialization and Normalization  
34 📄 [Scaling Rules for Training](<Lecture 34 - Scaling Rules for Training/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2034%20-%20Scaling%20Rules%20for%20Training/notebook.ipynb)  
35 📄 [A Hacker's Guide to Deep Learning](<Lecture 35 - A Hacker's Guide to Deep Learning/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2035%20-%20A%20Hacker%27s%20Guide%20to%20Deep%20Learning/notebook.ipynb)  

### Part VIII — Vision

36 📄 [How a Neural Network Learns to See: Convolution](<Lecture 36 - How a Neural Network Learns to See: Convolution/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2036%20-%20How%20a%20Neural%20Network%20Learns%20to%20See%3A%20Convolution/notebook.ipynb)  
37 ○ CNN Architectures: LeNet to ResNet  
38 📄 [Geometry, Invariance, and Equivariance](<Lecture 38 - Geometry, Invariance, and Equivariance/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2038%20-%20Geometry%2C%20Invariance%2C%20and%20Equivariance/notebook.ipynb)  

### Part IX — Sequences and Language

39 📄 [When Order Matters: Learning From Sequences](<Lecture 39 - When Order Matters: Learning From Sequences/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2039%20-%20When%20Order%20Matters%3A%20Learning%20From%20Sequences/notebook.ipynb)  
40 ○ LSTM and GRU: Fixing the Gradient  
41 📄 [How Can a Computer Represent the Meaning of a Word](<Lecture 41 - How Can a Computer Represent the Meaning of a Word/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2041%20-%20How%20Can%20a%20Computer%20Represent%20the%20Meaning%20of%20a%20Word/notebook.ipynb)  
42 📄 [Attention: What Should I Look At](<Lecture 42 - Attention: What Should I Look At/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2042%20-%20Attention%3A%20What%20Should%20I%20Look%20At/notebook.ipynb)  
43 📄 [Transformers: Building With Attention](<Lecture 43 - Transformers: Building With Attention/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2043%20-%20Transformers%3A%20Building%20With%20Attention/notebook.ipynb)  
44 📄 [How Does a Language Model Learn to Predict Text](<Lecture 44 - How Does a Language Model Learn to Predict Text/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2044%20-%20How%20Does%20a%20Language%20Model%20Learn%20to%20Predict%20Text/notebook.ipynb)  
45 📄 [Inference Methods](<Lecture 45 - Inference Methods/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2045%20-%20Inference%20Methods/notebook.ipynb)  

### Part X — Representation and Generation

46 📄 [Representation Learning](<Lecture 46 - Representation Learning/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2046%20-%20Representation%20Learning/notebook.ipynb)  
47 📄 [Contrastive Learning](<Lecture 47 - Contrastive Learning/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2047%20-%20Contrastive%20Learning/notebook.ipynb)  
48 📄 [How Can a Machine Create Something New](<Lecture 48 - How Can a Machine Create Something New/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2048%20-%20How%20Can%20a%20Machine%20Create%20Something%20New/notebook.ipynb)  
49 📄 [Conditional Generative Models](<Lecture 49 - Conditional Generative Models/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2049%20-%20Conditional%20Generative%20Models/notebook.ipynb)  
50 📄 [Graph Neural Networks](<Lecture 50 - Graph Neural Networks/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2050%20-%20Graph%20Neural%20Networks/notebook.ipynb)  

### Part XI — Theory and Scale

51 📄 [Why Neural Networks Generalize](<Lecture 51 - Why Neural Networks Generalize/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2051%20-%20Why%20Neural%20Networks%20Generalize/notebook.ipynb)  
52 📄 [Scaling Laws](<Lecture 52 - Scaling Laws/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2052%20-%20Scaling%20Laws/notebook.ipynb)  
53 📄 [Transfer Learning and Fine-Tuning](<Lecture 53 - Transfer Learning and Fine-Tuning/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2053%20-%20Transfer%20Learning%20and%20Fine-Tuning/notebook.ipynb)  
54 📄 [Out-of-Distribution and Robustness](<Lecture 54 - Out-of-Distribution and Robustness/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2054%20-%20Out-of-Distribution%20and%20Robustness/notebook.ipynb)  
55 📄 [Architectural Bias and Representations](<Lecture 55 - Architectural Bias and Representations/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2055%20-%20Architectural%20Bias%20and%20Representations/notebook.ipynb)  
56 📄 [Metrized Deep Learning](<Lecture 56 - Metrized Deep Learning/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2056%20-%20Metrized%20Deep%20Learning/notebook.ipynb)  

### Part XII — Frontier and Practice

57 📄 [Preference Learning and Policy Optimization](<Lecture 57 - Preference Learning and Policy Optimization/blog.md>) · [Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2057%20-%20Preference%20Learning%20and%20Policy%20Optimization/notebook.ipynb)  
58 ○ Retrieval, Tools and Agents  
59 ○ ML Systems: Serving, Quantization and Cost  
60 ○ From Course to Research  

---

**5 written · 42 drafted · 13 planned · 60 total**
