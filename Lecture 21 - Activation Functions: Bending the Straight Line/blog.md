# Lecture 21 — Activation Functions: Bending the Straight Line

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/main/Lecture%2021%20-%20Activation%20Functions%3A%20Bending%20the%20Straight%20Line/notebook.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2021%20-%20Activation%20Functions%3A%20Bending%20the%20Straight%20Line/notebook.ipynb)**

The blog is the textbook; the notebook is the laboratory. Predict the output first, then run the experiment.

## 🧭 Where this lesson fits

**Previous:** Lecture 20 — How Do We Know If Our Model Really Learned?

**Today:** We take the neuron $z=\mathbf w^T\mathbf x+b$ and ask why a neural network needs one more operation.

**Next:** Lecture 22 — Why Depth Makes Neural Networks Powerful.

The key bridge is:

> **A linear neuron can draw only a straight boundary. An activation function lets the network bend that boundary.**

---

## 1. The neuron we already know

A neuron first computes

$$
z=\mathbf w^T\mathbf x+b.
$$

With one input this becomes

$$
z=wx+b.
$$

That is an affine function: graph it, and you get a straight line.

For example,

$$
z=2x+1.
$$

Try $x=0,1,2,3$:

| $x$ | $z$ |
|---:|---:|
| 0 | 1 |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

Nothing surprising yet.

---

## 2. The strange problem with stacking layers

Imagine two layers with no activation function:

$$
\mathbf h=W_1\mathbf x+\mathbf b_1
$$

then

$$
\mathbf y=W_2\mathbf h+\mathbf b_2.
$$

Substitute the first equation into the second:

$$
\mathbf y=W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2.
$$

Distribute $W_2$:

$$
\mathbf y=W_2W_1\mathbf x+W_2\mathbf b_1+\mathbf b_2.
$$

The whole two-layer network is still of the form

$$
\mathbf y=W\mathbf x+\mathbf b.
$$

So two linear layers can collapse into **one** linear layer.

Three layers can collapse too. Ten can collapse too.

> ⚠️ **A tempting wrong idea:** “More layers automatically means a more powerful network.”
>
> Not if every layer is only linear. Without a nonlinear operation, depth gives us no new kind of function.

---

## 3. We need a bend

Suppose we want a rule like this:

```text
small input  → small output
middle input → almost nothing
large input  → large output
```

Or a classifier that says:

```text
left side  → class A
right side → class B
```

A straight line is sometimes too restrictive.

So after the weighted sum $z$, we apply another function:

$$
\boxed{a=f(z)}
$$

This $f$ is the **activation function**.

The neuron now becomes

$$
\boxed{a=f(\mathbf w^T\mathbf x+b)}.
$$

The weighted sum creates the raw signal.

The activation decides how that signal is transformed before the next layer sees it.

---

## 4. First activation: ReLU

A remarkably simple choice is **ReLU**:

$$
\boxed{f(z)=\max(0,z)}.
$$

In plain language:

```text
if z is negative → output 0
if z is positive → keep z
```

Examples:

| $z$ | ReLU$(z)$ |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

The graph has a bend at zero:

```text
output
  |
  |          /
  |         /
  |        /
--+-------/-------- input
  |
  |
```

That bend is the important part.

---

## 5. Why does a tiny bend matter so much?

Because when we stack layers, each layer can create new regions of the input space.

Imagine:

$$
\mathbf h=f(W_1\mathbf x+b_1)
$$

followed by

$$
\hat y=W_2\mathbf h+b_2.
$$

Now the second layer is no longer receiving a simple linear transformation of $\mathbf x$.

It is receiving the **bent, transformed representation** $\mathbf h$.

A later layer can combine those new pieces again.

That repeated process is what gives deep networks their expressive power.

---

## 6. Another activation: sigmoid

The **sigmoid** function is

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

It maps any real number into the interval $(0,1)$.

Some useful values:

| $z$ | $\sigma(z)$ approximately |
|---:|---:|
| -5 | 0.007 |
| -2 | 0.119 |
| 0 | 0.500 |
| 2 | 0.881 |
| 5 | 0.993 |

That can be useful when the output is interpreted as a probability-like score.

The shape looks like an S:

```text
1 |             ______
  |          __/
  |       __/
0 |______/____________ input
```

Unlike ReLU, sigmoid is smooth everywhere.

---

## 7. Tanh

Another classic activation is

$$
\boxed{\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}}
$$

It maps numbers into

$$
(-1,1).
$$

It is centered around zero:

| $z$ | $\tanh(z)$ approximately |
|---:|---:|
| -2 | -0.964 |
| -1 | -0.762 |
| 0 | 0 |
| 1 | 0.762 |
| 2 | 0.964 |

The important lesson is not to memorize three formulas blindly.

Instead ask:

> **What shape does this function create, and what happens to its derivative?**

---

## 8. Activation functions and gradients

Learning requires derivatives.

For a function $a=f(z)$, its derivative tells us how much the output changes when $z$ changes.

For ReLU, away from zero,

$$
\frac{da}{dz}=\begin{cases}
0, & z<0\\
1, & z>0
\end{cases}
$$

So a positive ReLU unit passes a gradient through unchanged, while a negative unit has zero local derivative.

For sigmoid,

$$
\boxed{\sigma'(z)=\sigma(z)(1-\sigma(z))}.
$$

Because sigmoid approaches 0 or 1 at large positive or negative inputs, its derivative becomes small there.

That fact will matter enormously when we study backpropagation.

---

## 9. A neuron is now a composition

Without activation:

$$
\mathbf x\rightarrow W\mathbf x+b
$$

With activation:

$$
\mathbf x\rightarrow W\mathbf x+b\rightarrow f(\cdot)
$$

In symbols,

$$
\boxed{\mathbf a=f(W\mathbf x+\mathbf b)}.
$$

This is the basic pattern repeated throughout neural networks.

A network is mostly a very large composition of operations like this.

---

## 10. A concrete example

Take

$$
\mathbf x=\begin{bmatrix}2\\-1\end{bmatrix},\quad
\mathbf w=\begin{bmatrix}3\\4\end{bmatrix},\quad b=-2.
$$

First compute the pre-activation:

$$
z=3(2)+4(-1)-2=0.
$$

Then with ReLU:

$$
a=\max(0,0)=0.
$$

Now change the bias from $-2$ to $1$:

$$
z=6-4+1=3
$$

and

$$
a=\max(0,3)=3.
$$

Same input. Same weights. Different bias. Different activation output.

---

## 11. Build the activation yourself in Python

```python
import numpy as np

z = np.array([-3., -1., 0., 2., 5.])

relu = np.maximum(0, z)
print(relu)
```

Sigmoid:

```python
sigmoid = 1 / (1 + np.exp(-z))
print(sigmoid)
```

Try changing the values in `z` before looking at the result.

---

## 12. The same idea in PyTorch

```python
import torch
import torch.nn as nn

z = torch.tensor([-3., -1., 0., 2., 5.])

relu = nn.ReLU()
sigmoid = nn.Sigmoid()

print(relu(z))
print(sigmoid(z))
```

The library is convenient, but the mathematics is still just function evaluation.

---

## 13. The hidden layer gets its name for a reason

Suppose

$$
\mathbf h=\operatorname{ReLU}(W_1\mathbf x+\mathbf b_1).
$$

We usually do not call $\mathbf h$ the answer.

It is an internal representation created by the network.

That is why these units are often called a **hidden layer**.

The network learns which intermediate features are useful for solving the final task.

---

## 🎮 Interactive activation playground

Change the activation function while keeping the same $z$ values.

Try to predict:

1. What happens to negative numbers under ReLU?
2. Which activation outputs exactly zero for $z<0$?
3. Which activations are bounded?
4. Where is sigmoid's derivative largest?
5. Which activation is centered around zero?

A useful visualization is:

$$
z\rightarrow f(z)$$

with the graph and derivative shown together.

---

## ⚠️ Common misconceptions

### “Activation functions make the network learn.”
Not by themselves. They make nonlinear function classes available. Learning still requires an objective and parameter updates.

### “ReLU is just a coding trick.”
No. Its shape changes the mathematical function represented by the network.

### “Any nonlinear activation gives the same result.”
No. Different activations have different ranges, smoothness, derivatives, and optimization behavior.

### “A negative ReLU output is just a little smaller.”
No. It is exactly zero.

---

## 🔬 Failure mode: all-linear networks

Consider

$$
\mathbf h=W_1\mathbf x+b_1
$$

and

$$
\hat y=W_2\mathbf h+b_2.
$$

Even though this looks like a two-layer network, it collapses to one affine transformation.

So the failure is not “too few parameters.”

The failure is **missing nonlinearity**.

---

## 🧩 Exercises

**Level 1 — Calculate**

1. Compute ReLU for $[-4,-1,0,2,7]$.
2. Compute $z$ for $x=[2,-3]$, $w=[4,2]$, $b=-1$, then apply ReLU.

**Level 2 — Explain**

3. Why can two affine layers be replaced by one affine layer?
4. Why does an activation function prevent that collapse?

**Level 3 — Investigate**

5. Plot ReLU, sigmoid and tanh on the same input range.
6. Plot each derivative and identify where the derivative becomes small.

**Level 4 — Research bridge**

7. Why can dead ReLU units become a practical optimization problem?
8. Why is GELU widely used in modern transformer architectures?

---

## 🏁 Mastery gate

Move to Lecture 22 only if you can:

- explain why affine layers alone collapse into one affine transformation;
- calculate ReLU, sigmoid and tanh for simple values;
- explain why nonlinearity is the key role of an activation function;
- connect activation derivatives to later gradient calculations;
- write the forward equation $\mathbf a=f(W\mathbf x+\mathbf b)$;
- explain why hidden representations are useful.

---

## What you should remember

> **A neuron becomes much more powerful when we insert a nonlinear function after the weighted sum.**

The core equation is

$$
\boxed{\mathbf a=f(W\mathbf x+\mathbf b)}.
$$

Without $f$, depth can collapse into one linear transformation.

With $f$, layers can build progressively more complicated functions.

**Next:** we stack these nonlinear layers and see why depth changes what a network can represent.
