# Blog 05 — Meet the Smallest Neural Network

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/main/Lecture%2005%20-%20Meet%20the%20Smallest%20Neural%20Network/notebook.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2005%20-%20Meet%20the%20Smallest%20Neural%20Network/notebook.ipynb)**

The blog is the textbook; the notebook is the laboratory. Run every experiment and predict its result before executing it.

## 🧭 Where this lesson fits

**Came from:** Blog 04 — A Matrix Can Transform Space.

**Today:** We turn a matrix transformation into the basic building block of a neural network: a neuron.

**Next:** Blog 06 — Why Does a Neuron Need an Activation Function?

The key bridge is simple:

> **A neuron is a mathematical function that takes numbers in, combines them using learned parameters, and produces a number out.**

---

## 1. Start with one input

Imagine a student studies for `x` hours and we want a very simple score prediction.

A model could be:

$$
\hat y = wx
$$

If $x=3$ and $w=2$:

$$
\hat y=2\times3=6.
$$

This is already a model. It is not learning yet; it is simply applying a rule.

### What is the weight?

The number $w$ tells us how strongly the input affects the output of this model.

- $w>0$: increasing $x$ increases the output.
- $w<0$: increasing $x$ decreases the output.
- $w=0$: the input has no effect in this calculation.

A learned weight is a model parameter, **not automatically a causal explanation**.

---

## 2. Give the neuron a bias

Real relationships often do not pass through zero. So we add a bias:

$$
\boxed{\hat y=wx+b}
$$

For

$$
w=2,\quad x=3,\quad b=1,
$$

we get

$$
\hat y=2(3)+1=7.
$$

Think of the weight as controlling the **tilt** of the line and the bias as moving the line **up or down**.

For $\hat y=2x+1$:

```text
prediction
  9 |             ●
  7 |          ●
  5 |       ●
  3 |    ●
  1 | ●
    +------------------ input
      0  1  2  3  4
```

So the smallest neuron begins as a straight line.

---

## 3. Why this connects to Blog 04

In the previous lesson, a matrix transformed vectors.

A neuron performs the same basic kind of operation, but then reduces the transformed information to a single number:

$$
\mathbf{x}\rightarrow\mathbf{w}^T\mathbf{x}+b.
$$

The dot product is the bridge between linear algebra and neural networks.

---

## 4. Give the neuron several inputs

Suppose our student has three features:

$$
\mathbf{x}=\begin{bmatrix}2\\3\\4\end{bmatrix}
$$

and the neuron has weights

$$
\mathbf{w}=\begin{bmatrix}4\\5\\2\end{bmatrix},\qquad b=1.
$$

The neuron calculates

$$
z=\mathbf{w}^T\mathbf{x}+b.
$$

Step by step:

$$
z=4(2)+5(3)+2(4)+1
$$

$$
z=8+15+8+1=32.
$$

That is all a basic artificial neuron needs before its activation function.

### The dot product as a conversation

Each input says, “Here is my value.”

Each weight says, “Here is how strongly I count.”

The neuron adds all those contributions and adds the bias.

$$
\boxed{z=\sum_i w_i x_i+b}
$$

---

## 5. What does a negative weight do?

Try

$$
\mathbf{x}=\begin{bmatrix}2\\3\end{bmatrix},\quad
\mathbf{w}=\begin{bmatrix}4\\-5\end{bmatrix},\quad b=1.
$$

Then

$$
z=4(2)-5(3)+1=-6.
$$

The second feature pushes the result downward.

This is a useful mental model:

```text
x₁ ──× w₁ ──┐
             ├── add ── + b ── z
x₂ ──× w₂ ──┘
```

The multiplication creates each feature's contribution; addition combines them.

---

## 6. From one neuron to a layer

One neuron produces one output. What if we want three outputs?

Use three neurons at once:

$$
\mathbf{z}=W\mathbf{x}+\mathbf{b}.
$$

For

$$
W=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix},\qquad
\mathbf{x}=\begin{bmatrix}2\\3\end{bmatrix},
$$

we get

$$
W\mathbf{x}=
\begin{bmatrix}
1(2)+2(3)\\
3(2)+4(3)\\
5(2)+6(3)
\end{bmatrix}
=
\begin{bmatrix}8\\18\\28\end{bmatrix}.
$$

Each row of $W$ is one neuron's weight vector.

This is why matrix multiplication is so important in deep learning: **many neuron calculations become one matrix operation**.

---

## 7. Count the parameters

For a neuron with $d$ inputs:

- $d$ weights
- 1 bias

So it has

$$
\boxed{d+1}
$$

learnable parameters.

For a layer with $d$ inputs and $m$ neurons:

$$
\boxed{md+m=m(d+1)}
$$

parameters.

For example, 4 inputs and 3 neurons require

$$
3(4)+3=15
$$

parameters.

Parameter counting becomes extremely important later when we study model size, memory, compute, and scaling laws.

---

## 8. Parameters versus data

Keep these roles separate:

| Symbol | Meaning | Changes during training? |
|---|---|---|
| $x$ | input data | No, for a fixed example |
| $w$ | learned weight | Yes |
| $b$ | learned bias | Yes |
| $z$ | weighted sum | Changes when inputs/parameters change |
| $\hat y$ | prediction | Changes when inputs/parameters change |

Training does **not** normally change the training examples themselves. It changes the parameters so the same model becomes better at the task.

---

## 9. Why call it a neuron?

The terminology comes from biological inspiration, but an artificial neuron is not a realistic simulation of a biological cell.

A simplified artificial neuron is:

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

and, in a modern neural network,

$$
a=f(z).
$$

The function $f$ is an activation function. We will study why it is necessary next.

---

## 10. Build the neuron yourself in NumPy

```python
import numpy as np

x = np.array([2.0, 3.0, 4.0])
w = np.array([4.0, 5.0, 2.0])
b = 1.0

z = w @ x + b
print(z)  # 32.0
```

The `@` operator means matrix/vector multiplication. Here it computes the dot product.

There is no neural-network magic hidden in this calculation:

**multiply → add → produce a number.**

---

## 11. The same neuron in PyTorch

```python
import torch

x = torch.tensor([2., 3., 4.])
w = torch.tensor([4., 5., 2.])
b = torch.tensor(1.)

z = w @ x + b
print(z)
```

PyTorch gives us the same mathematics plus automatic differentiation, optimizers, GPUs, and neural-network building blocks.

A good learning rule is:

> **Understand the arithmetic first. Use PyTorch second.**

---

## 12. A batch of examples

Suppose several examples arrive together as rows of a matrix:

$$
X=\begin{bmatrix}
1&2\\
2&3\\
3&4
\end{bmatrix}.
$$

For a single-output neuron with

$$
\mathbf{w}=\begin{bmatrix}2\\3\end{bmatrix},\qquad b=1,
$$

we can calculate all predictions at once:

$$
\hat{\mathbf y}=X\mathbf w+b.
$$

Here:

$$
X\mathbf w=
\begin{bmatrix}
8\\13\\18
\end{bmatrix}
$$

so

$$
\hat{\mathbf y}=\begin{bmatrix}9\\14\\19\end{bmatrix}.
$$

This is the beginning of **vectorized computing**: instead of writing a loop for every example, we express the whole batch as matrix multiplication.

---

## 13. The neuron is still not learning

Suppose we start with

$$
w=0,\qquad b=0.
$$

Then

$$
\hat y=0
$$

for every input.

The neuron can calculate, but it does not know whether its answer is good.

That gives us the next question:

> **How can the machine measure its mistake and decide how to change $w$ and $b$?**

That is the transition from **forward calculation** to **learning**.

---

## 🎮 Interactive neuron playground

The matching notebook should let you change one parameter at a time and observe the output.

Try these experiments:

1. Keep $w$ fixed and increase $b$. Does the line move or rotate?
2. Keep $b$ fixed and increase $w$. Does the line move or rotate?
3. Set one weight to zero. Which feature disappears from the calculation?
4. Make one weight negative. Which direction does its contribution move?
5. Add a third input and predict the new output before running the code.

For a browser implementation, visualize:

$$
x_i\xrightarrow{\times w_i}w_ix_i\xrightarrow{\text{sum}}z\xrightarrow{+b}z+b.
$$

The static equations above remain the source of truth even when the playground is unavailable.

---

## 🧠 Think like a scientist

Take

$$
\hat y=3x-2.
$$

Calculate the predictions for $x=0,1,2,3$.

Then change $w=3$ to $w=1$.

- What changed?
- Did the intercept change?
- Did the slope change?

Now change $b=-2$ to $b=4$.

- What changed this time?

Do not just run the notebook. **Predict first, then test.**

---

## ⚠️ Common misconceptions

### “A neuron is a tiny brain.”
No. It is a mathematical function inspired loosely by biology.

### “A large weight means the feature is important in the real world.”
Not necessarily. Feature scale, interactions, regularization, and model structure all matter.

### “Matrix multiplication is separate from neural networks.”
No. Dense neural-network layers are fundamentally matrix multiplication plus bias, followed by an activation in typical architectures.

### “If a neuron makes a prediction, it has learned.”
No. Prediction is the forward computation. Learning requires an objective and a parameter-update mechanism.

---

## 🔬 Failure mode

A neuron with only

$$
\hat y=\mathbf w^T\mathbf x+b
$$

can represent only linear/affine relationships.

That is powerful but limited. Many real patterns are curved, conditional, or otherwise nonlinear.

The next lesson asks the crucial question:

> **What happens when we insert a nonlinear function after the weighted sum?**

That is why activation functions exist.

---

## 🧩 Exercises

**Level 1 — Calculate**

1. Compute $z$ for $x=[2,5]$, $w=[3,-1]$, $b=4$.
2. Compute the three outputs for a $2\times3$ weight matrix and a 3-element input vector.

**Level 2 — Explain**

3. Explain the difference between a weight and a bias using a line graph.
4. Why can three neurons be represented by one matrix multiplication?

**Level 3 — Investigate**

5. Build a neuron with 5 inputs and count its parameters.
6. Compare predictions before and after changing only one weight.

**Level 4 — Research bridge**

7. Why does adding nonlinear activation make a deep network much more expressive than stacking affine transformations alone?
8. Investigate how parameter count affects memory and compute in a dense layer.

---

## 🏁 Mastery gate

Move to Blog 06 only if you can:

- calculate $\mathbf w^T\mathbf x+b$ by hand;
- explain what a weight and bias do geometrically;
- write the NumPy implementation;
- express multiple neurons as $W\mathbf x+\mathbf b$;
- count parameters in a dense layer;
- explain why prediction is not yet learning;
- predict what happens when one parameter changes.

---

## What you should remember

> **A neural network begins with a very simple idea: multiply inputs by weights, add them together, add a bias, and produce a number.**

The core calculation is

$$
\boxed{z=\mathbf w^T\mathbf x+b}.
$$

Everything that looks spectacular later—deep networks, transformers, language models—builds on increasingly sophisticated compositions of simple mathematical operations.

**Next:** we add nonlinearity and discover why simply stacking straight lines is not enough.
