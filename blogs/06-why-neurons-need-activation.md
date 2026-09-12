# Blog 06 — Why Does a Neuron Need an Activation Function?

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/reorg/class8-to-phd-curriculum/notebooks/06-why-neurons-need-activation.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/reorg/class8-to-phd-curriculum/notebooks/06-why-neurons-need-activation.ipynb)**

## 🧭 Where this lesson fits

**Came from:** Blog 05 — a neuron computes $z=\mathbf w^T\mathbf x+b$.

**Today:** discover why a second function $a=f(z)$ is essential.

**Next:** Blog 07 — prediction is not the same as learning.

---

## 1. The surprising problem with many layers

You might think that if one line is useful, ten layers of lines must be incredibly powerful.

But consider

$$f(x)=2x+1$$

and

$$g(x)=3x-4.$$

Stack them:

$$g(f(x))=3(2x+1)-4=6x-1.$$

It is still just a line.

In general, composing affine functions gives another affine function:

$$W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2).$$

So **depth without nonlinearity does not buy us the expressive power we expect from a deep network**.

---

## 2. Add an activation

A neuron becomes

$$
\boxed{z=\mathbf w^T\mathbf x+b,\qquad a=f(z)}
$$

or, in one equation,

$$
\boxed{a=f(\mathbf w^T\mathbf x+b)}.
$$

The activation function bends, clips, gates, or otherwise transforms the weighted sum.

That small extra operation changes what a network can represent.

---

## 3. ReLU: the simplest example

ReLU means **Rectified Linear Unit**:

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

Examples:

| $z$ | ReLU$(z)$ |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

So ReLU behaves like a switch:

- negative input → output $0$;
- positive input → pass the value through.

```text
output
  |
  |        /
  |       /
  |      /
  |_____/________ input
       0
```

It is simple, but it makes a network **piecewise linear** rather than one single line.

---

## 4. See the bend numerically

Compare

$$f(x)=2x+1$$

with

$$g(x)=\operatorname{ReLU}(2x+1).$$

For $x=-2,-1,0,1,2$:

| $x$ | $2x+1$ | ReLU$(2x+1)$ |
|---:|---:|---:|
| -2 | -3 | 0 |
| -1 | -1 | 0 |
| 0 | 1 | 1 |
| 1 | 3 | 3 |
| 2 | 5 | 5 |

The function now behaves differently in two regions.

That is the first glimpse of how networks create complicated shapes from simple pieces.

---

## 5. Other important activations

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Its output lies strictly between 0 and 1. It is useful in some output layers, especially binary classification, though a sigmoid output is not automatically a calibrated probability.

### Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}.
$$

Its output lies between $-1$ and $1$.

### ReLU

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

Modern networks also use variants such as GELU, SiLU/Swish, and Leaky ReLU. Later we will compare their shapes and derivatives.

---

## 6. Why nonlinear functions create richer shapes

Imagine one ReLU neuron creates a bend at some threshold. Several neurons can create several bends. A later layer can combine those pieces.

Conceptually:

```text
linear layer → nonlinearity → linear layer → nonlinearity → ...
```

This lets a network construct functions that a single affine transformation cannot represent.

The famous universal-approximation results are more subtle than “one hidden layer can learn everything”: they depend on architecture, activation, width, approximation domain, and assumptions. The important beginner lesson is simply that **nonlinearity is what lets composition become genuinely richer**.

---

## 7. Activation is not learning

An activation function does not learn by itself.

It is a fixed mathematical operation unless we explicitly make some of its parameters learnable.

Learning still requires:

$$
\text{prediction}\rightarrow\text{loss}\rightarrow\text{gradient}\rightarrow\text{parameter update}.
$$

We will meet that missing machinery in the next lessons.

---

## 8. NumPy: write ReLU yourself

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

z = np.array([-3., -1., 0., 2., 5.])
print(relu(z))
```

Expected output:

```text
[0. 0. 0. 2. 5.]
```

Do not hide this behind a library yet. Writing the five-line version makes the idea concrete.

---

## 9. PyTorch verification

```python
import torch

z = torch.tensor([-2., -1., 0., 1., 2.])
print(torch.relu(z))
```

PyTorch gives the same result and integrates the operation with automatic differentiation.

---

## 🎮 Activation playground

In the notebook, change the activation and observe its graph.

Try:

1. ReLU
2. sigmoid
3. tanh
4. Leaky ReLU
5. GELU

For each one ask:

- What is the output range?
- Is it smooth?
- What happens for a very large positive input?
- What happens for a very large negative input?
- What might its derivative look like?

The derivative questions will become important when we study backpropagation.

---

## ⚠️ Failure mode: dead ReLU

For ordinary ReLU, if a neuron receives negative pre-activations for all relevant examples, its output is always zero and its derivative is zero there. A neuron can therefore become difficult to update through that path—the familiar **dying ReLU** issue.

This is one reason alternative activations exist.

---

## 🧠 Common misconceptions

**“More layers automatically means more power.”**

Not if every layer is only affine; their composition collapses to another affine transformation.

**“ReLU removes information permanently, so it must always be bad.”**

ReLU does discard negative values at that activation, but this controlled gating is often useful. Whether information loss is harmful depends on the representation and task.

**“Sigmoid is always the best activation.”**

No. Its saturation can produce very small derivatives for large positive or negative inputs, which can make optimization harder in deep hidden networks.

---

## 🔬 Scientist experiment

Plot ReLU, sigmoid and tanh on the same axis.

Then estimate their slopes numerically near $z=0$ and far from zero.

Make a hypothesis:

> Which activation should make gradient-based learning easier in a deep hidden layer, and why?

Do not trust the first answer you find. Test the functions and then connect the observation to derivatives.

---

## 🧩 Exercises

1. Compute ReLU for `[-4,-1,0,2,7]` by hand.
2. Explain why two affine layers can collapse into one affine layer.
3. Draw a piecewise-linear function made by combining two ReLU units.
4. Compare sigmoid and tanh output ranges.
5. Numerically estimate the derivative of sigmoid at $z=0$.
6. Research why ReLU became popular despite being non-differentiable exactly at zero.

### Research bridge

Study the relationship between activation choice, gradient propagation, optimization speed, and representation quality. Compare at least two activations under the same architecture, seed, optimizer, learning rate, and dataset.

---

## 🏁 Mastery gate

Move on when you can:

- explain why stacked affine layers remain affine;
- calculate ReLU by hand;
- draw ReLU, sigmoid and tanh;
- explain why nonlinearity changes expressiveness;
- implement an activation in NumPy;
- verify it in PyTorch;
- describe one activation failure mode;
- predict how changing an activation changes a graph.

## What you should remember

> **The neuron gives us a weighted sum. The activation gives the network the ability to bend that sum into richer functions.**

$$
\boxed{a=f(\mathbf w^T\mathbf x+b)}
$$

**Next:** the network can now make predictions. But how will it know whether those predictions are good?