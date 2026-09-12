# Lecture 24 — Meet the Smallest Neural Network

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/main/Lecture%2024%20-%20Meet%20the%20Smallest%20Neural%20Network/notebook.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2024%20-%20Meet%20the%20Smallest%20Neural%20Network/notebook.ipynb)**

The blog is the textbook; the notebook is the laboratory. Predict first, run second, explain third.

## 🧭 Where We Are

**Previously:** We learned that vectors let us keep many measurements together and that dot products turn matching numbers into one useful score.

**Today:** We build the smallest useful neural-network unit: a neuron. We will discover that its first job is not mysterious at all—it is a weighted sum plus a bias.

**Next:** A straight-line neuron is too limited. We will add an activation function and see why nonlinearity is the ingredient that makes depth useful.

> **Big Question:** How can a handful of numbers be combined into a useful prediction?

---

## 1. The Problem: One Number In, One Number Out

Imagine a teacher wants to predict a student's exam score from study time.

| Study hours $x$ | Score $y$ |
|---:|---:|
| 1 | 42 |
| 2 | 54 |
| 3 | 66 |
| 4 | 78 |

A tiny model could be

$$
\hat y = wx.
$$

But at $x=0$, this model is forced to predict $0$. Real relationships often do not behave like that.

**The model needs a second adjustable number.**

---

## 2. What Would a Useful Neuron Need?

A tiny prediction unit should:

1. let each input have its own influence;
2. combine those influences into one number;
3. allow the whole output to shift up or down;
4. work with one input today and many inputs later;
5. expose numbers that learning can change.

---

## 3. First Attempt: Just Multiply

Start with

$$
\hat y=wx.
$$

Suppose the real rule is approximately

$$
\hat y=12x+30.
$$

Try $w=12$:

| $x$ | Prediction $12x$ | Truth |
|---:|---:|---:|
| 1 | 12 | 42 |
| 2 | 24 | 54 |
| 3 | 36 | 66 |
| 4 | 48 | 78 |

The slope is right, but the whole line is 30 points too low.

> ⚠️ **A tempting wrong idea**
>
> “Just change the weight more.”
>
> Changing $w$ rotates the line around zero. It cannot independently move the line upward. We need a second dial.

---

## 4. The Discovery: Add a Bias

Add a constant term:

$$
\boxed{\hat y=wx+b}
$$

Choose $w=12$ and $b=30$:

$$
\hat y=12(3)+30=66.
$$

| Parameter | Intuition | Geometric effect |
|---|---|---|
| $w$ | how strongly input matters | changes slope / tilt |
| $b$ | starting offset | shifts the line up or down |

> 💡 **Core idea**
>
> A neuron is not a tiny brain. It is a mathematical function with parameters that can be adjusted.

---

## 5. Give the Neuron More Inputs

Suppose a student's prediction depends on study hours, attendance, and practice tests:

$$
\mathbf{x}=\begin{bmatrix}2\\80\\3\end{bmatrix},
\qquad
\mathbf{w}=\begin{bmatrix}10\\0.2\\4\end{bmatrix},
\qquad b=5.
$$

The neuron computes

$$
z=\mathbf{w}^T\mathbf{x}+b.
$$

Step by step:

$$
z=10(2)+0.2(80)+4(3)+5=20+16+12+5=53.
$$

There is no new magic: **multiply each input by its weight, add the results, then add the bias.**

---

## 6. The Dot Product Is the Bridge

From Chapter 2:

$$
\mathbf{w}^T\mathbf{x}=\sum_{i=1}^{d}w_ix_i.
$$

So the neuron is simply

$$
\boxed{z=\mathbf{w}^T\mathbf{x}+b}.
$$

The one-input model $wx+b$ is just the special case $d=1$.

> 🧠 **Think**
>
> The input says, “Here are my measurements.” Each weight says, “Here is how strongly I count.” The dot product adds those contributions. The bias shifts the score.

---

## 7. What Does a Negative Weight Mean?

Take

$$
\mathbf{x}=\begin{bmatrix}2\\3\end{bmatrix},
\quad
\mathbf{w}=\begin{bmatrix}4\\-5\end{bmatrix},
\quad b=1.
$$

Then

$$
z=4(2)-5(3)+1=-6.
$$

```text
x₁ ──× w₁ ──┐
             ├── add ── + b ── z
x₂ ──× w₂ ──┘
```

A learned weight is an algebraic influence, not automatically a causal explanation.

---

## 8. From One Neuron to a Layer

Use three neurons at once:

$$
W=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix},
\qquad
\mathbf{x}=\begin{bmatrix}2\\3\end{bmatrix}.
$$

Then

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

With a bias vector,

$$
\boxed{\mathbf{z}=W\mathbf{x}+\mathbf{b}}.
$$

Each row of $W$ is one neuron's weight vector. Many neuron calculations become one matrix operation.

---

## 9. Count the Parameters

A neuron with $d$ inputs has $d$ weights and one bias:

$$
\boxed{d+1}.
$$

A layer with $d$ inputs and $m$ neurons has

$$
\boxed{md+m=m(d+1)}.
$$

For 4 inputs and 3 neurons:

$$
3(4)+3=15.
$$

---

## 10. Parameters Are Not the Data

| Symbol | Meaning | Changes during training? |
|---|---|---|
| $\mathbf{x}$ | input example | No, for a fixed example |
| $\mathbf{w}$ / $W$ | learned weights | Yes |
| $b$ / $\mathbf{b}$ | learned bias | Yes |
| $z$ / $\mathbf{z}$ | weighted sum | When inputs or parameters change |
| $\hat y$ | prediction | When inputs or parameters change |

Training changes model parameters. It does not rewrite the training examples.

---

## 11. A Batch of Examples

Let

$$
X=\begin{bmatrix}1&2\\2&3\\3&4\end{bmatrix},
\qquad
\mathbf{w}=\begin{bmatrix}2\\3\end{bmatrix},
\qquad b=1.
$$

All three predictions can be computed together:

$$
\hat{\mathbf y}=X\mathbf w+b.
$$

Since

$$
X\mathbf w=\begin{bmatrix}8\\13\\18\end{bmatrix},
$$

we get

$$
\hat{\mathbf y}=\begin{bmatrix}9\\14\\19\end{bmatrix}.
$$

This is the beginning of vectorized computing.

---

## 12. The Neuron Still Has Not Learned

Suppose

$$
\mathbf w=\mathbf0,\qquad b=0.
$$

Then every input gives $z=0$.

The neuron can calculate, but it cannot tell whether zero is good.

> **Forward computation is not learning.**

Learning needs a loss, gradients, and parameter updates.

---

## 13. Build It in NumPy

```python
import numpy as np

x = np.array([2.0, 3.0, 4.0])
w = np.array([4.0, 5.0, 2.0])
b = 1.0

z = w @ x + b
print(z)  # 32.0
```

There is no hidden neural-network spell:

**multiply → add → shift.**

---

## 14. Verify in PyTorch

```python
import torch

x = torch.tensor([2., 3., 4.])
w = torch.tensor([4., 5., 2.])
b = torch.tensor(1.)

z = w @ x + b
print(z)
```

PyTorch performs the same mathematics while providing automatic differentiation and other training tools.

> **Understand the arithmetic first. Use frameworks second.**

---

## 🎮 Interactive Neuron Playground

Change one quantity at a time and predict first.

1. Increase $b$ while keeping $w$ fixed. Does the line move or rotate?
2. Increase $w$ while keeping $b$ fixed. What changes?
3. Set one weight to zero. Which feature disappears?
4. Make one weight negative. Which contribution changes direction?
5. Add a third input. Can you predict the dot product before running the cell?

Visualize

$$
x_i\xrightarrow{\times w_i}w_ix_i\xrightarrow{\text{sum}}\mathbf{w}^T\mathbf{x}\xrightarrow{+b}z.
$$

---

## 🧠 Think Like a Scientist

For

$$
\hat y=3x-2,
$$

calculate predictions for $x=0,1,2,3$.

Then change $w=3$ to $w=1$.

- What changed?
- Did the intercept change?
- Did the slope change?

Now change $b=-2$ to $b=4$.

Predict before you run the notebook.

---

## ⚠️ Common Misconceptions

### “A neuron is a tiny brain.”
No. It is a mathematical function inspired loosely by biological terminology.

### “A big weight means the feature is important in the real world.”
Not necessarily. Feature scale and correlations matter.

### “A neuron that makes a prediction has learned.”
No. Prediction is computation. Learning requires an objective and parameter updates.

### “Matrix multiplication is unrelated to neural networks.”
Dense layers are fundamentally matrix multiplication plus bias, followed by an activation in typical architectures.

---

## 🔬 Failure Mode: The Straight-Line Trap

A neuron with only

$$
z=\mathbf w^T\mathbf x+b
$$

is an affine function. It cannot represent arbitrary curved or conditional relationships.

So the next question is unavoidable:

> **What happens when we put a nonlinear function after the weighted sum?**

That is the role of the activation function.

---

## 🧩 Exercises

**Level 1 — Calculate**

1. Compute $z$ for $\mathbf x=[2,5]$, $\mathbf w=[3,-1]$, $b=4$.
2. A neuron has 7 inputs. How many trainable parameters does it contain?

**Level 2 — Explain**

3. Explain weight versus bias using the graph of $y=wx+b$.
4. Why can three neurons be represented by one matrix multiplication?

**Level 3 — Investigate**

5. Build a neuron with 5 inputs and change only one weight.
6. Build a two-neuron layer and verify that each row of $W$ produces one output.

**Level 4 — Research Bridge**

7. Why can an affine model not represent an XOR decision boundary?
8. How does parameter count affect memory and compute in a dense layer?

---

## 🏁 Mastery Gate

Move on only if you can:

- calculate $\mathbf w^T\mathbf x+b$ by hand;
- explain what a weight and bias do;
- connect a dot product to a neuron;
- express multiple neurons as $W\mathbf x+\mathbf b$;
- count dense-layer parameters;
- distinguish data, parameters, pre-activation, and prediction;
- explain why forward computation is not yet learning.

## What You Should Remember

> **A neural network begins with a simple calculation: multiply each input by a learned weight, add the contributions, then add a bias.**

$$
\boxed{z=\mathbf w^T\mathbf x+b}
$$

Everything spectacular later—deep networks, transformers, language models—builds on increasingly sophisticated compositions of simple mathematical operations.

**Next:** we add nonlinearity and discover why stacking straight lines alone is not enough.
