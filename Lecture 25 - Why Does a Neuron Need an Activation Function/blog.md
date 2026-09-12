# Lecture 25 — Why Does a Neuron Need an Activation Function?

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/main/Lecture%2025%20-%20Why%20Does%20a%20Neuron%20Need%20an%20Activation%20Function/notebook.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2025%20-%20Why%20Does%20a%20Neuron%20Need%20an%20Activation%20Function/notebook.ipynb)**

The blog is the textbook; the notebook is the laboratory. Predict the graph before you run it.

## 🧭 Where We Are

**Previously:** We built a neuron that computes a weighted sum plus a bias.

**Today:** We discover a surprising limitation: stacking many affine layers still produces only an affine function. We then introduce the activation function.

**Next:** We use nonlinear neurons to build richer functions and connect the idea to function approximation.

> **Big Question:** Why does a deep network need a nonlinear activation function at all?

---

## 1. The Problem: Why Not Just Stack More Neurons?

Suppose one neuron computes

$$
f(x)=2x+1.
$$

Another layer computes

$$
g(x)=3x-4.
$$

Surely using both must make the model much more powerful.

Let's calculate:

$$
g(f(x))=3(2x+1)-4=6x-1.
$$

It is still a straight line.

That is the surprise.

---

## 2. What Would a Deep Network Need?

If adding layers is supposed to give us more expressive power, the layers need to be able to create something that one affine transformation cannot.

We want a building block that can:

1. transform a number;
2. behave differently in different input regions;
3. be composed repeatedly without collapsing back into one straight line;
4. remain simple enough to compute efficiently.

A linear or affine transformation satisfies only the first requirement.

---

## 3. First Attempt: Stack Affine Layers Forever

Consider two matrix layers:

$$
\mathbf h=W_1\mathbf x+\mathbf b_1,
$$

followed by

$$
\mathbf y=W_2\mathbf h+\mathbf b_2.
$$

Substitute the first equation into the second:

$$
\mathbf y=W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2.
$$

Distribute:

$$
\mathbf y=(W_2W_1)\mathbf x+(W_2\mathbf b_1+\mathbf b_2).
$$

That has exactly the form

$$
\mathbf y=W\mathbf x+\mathbf b.
$$

So two affine layers collapse into one affine layer.

The same argument works for three, ten, or a thousand affine layers.

> ⚠️ **A tempting wrong idea**
>
> “More layers automatically mean a much more powerful model.”
>
> Not when every layer is affine. Depth alone does not create nonlinearity.

---

## 4. The Discovery: Insert a Nonlinearity

After the weighted sum, apply a function:

$$
\boxed{z=\mathbf w^T\mathbf x+b}
$$

then

$$
\boxed{a=f(z)}.
$$

Together:

$$
\boxed{a=f(\mathbf w^T\mathbf x+b)}.
$$

The new ingredient is $f$.

A nonlinear $f$ prevents a stack of layers from collapsing into one affine transformation.

> 💡 **Core idea**
>
> The weighted sum gives the neuron a direction and offset. The activation changes the shape of the function.

---

## 5. ReLU: Our First Nonlinear Activation

The simplest important example is the **Rectified Linear Unit**:

$$
\boxed{\operatorname{ReLU}(z)=\max(0,z)}.
$$

Compute a few values:

| $z$ | ReLU$(z)$ |
|---:|---:|
| -3 | 0 |
| -1 | 0 |
| 0 | 0 |
| 2 | 2 |
| 5 | 5 |

It acts like a switch:

- negative input → output 0;
- positive input → pass the value through.

Graphically:

```text
output
  |
  |        /
  |       /
  |      /
  |_____/________ input
       0
```

The function is not one straight line. It has two regions joined at zero.

---

## 6. See the Bend With Numbers

Start with the affine function

$$
f(x)=2x+1.
$$

Now apply ReLU:

$$
g(x)=\operatorname{ReLU}(2x+1).
$$

| $x$ | $2x+1$ | ReLU$(2x+1)$ |
|---:|---:|---:|
| -2 | -3 | 0 |
| -1 | -1 | 0 |
| 0 | 1 | 1 |
| 1 | 3 | 3 |
| 2 | 5 | 5 |

The affine function keeps decreasing below zero. ReLU clips that entire region to zero.

This creates a **piecewise-linear** function.

That phrase sounds advanced, but it only means:

> **One simple rule on one side, another simple rule on the other side.**

---

## 7. Many ReLU Neurons Can Make Many Bends

Imagine one ReLU makes one bend.

Several hidden neurons can make several bends:

```text
input
  ↓
weighted sums
  ↓
ReLU ReLU ReLU ReLU
  ↓
combine
  ↓
output
```

A later layer can combine these simple pieces into a more complicated curve.

This is the basic geometric reason nonlinear networks can model complicated patterns: **they combine many simple local behaviors.**

---

## 8. Other Activations

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Its output is strictly between 0 and 1.

For example, approximately:

| $z$ | sigmoid$(z)$ |
|---:|---:|
| -5 | 0.0067 |
| 0 | 0.5 |
| 5 | 0.9933 |

Sigmoid is useful in some output layers, especially binary classification, but a sigmoid output is not automatically a calibrated probability.

### Tanh

$$
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}.
$$

Its output lies between $-1$ and $1$.

### ReLU

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

Modern architectures also use variants such as Leaky ReLU, GELU, and SiLU/Swish.

The important first step is not memorizing names. It is understanding what the shape of the function does.

---

## 9. Why Nonlinearity Matters More Than Depth Alone

Compare these two networks:

### Network A — affine only

$$
\mathbf x\rightarrow W_1\mathbf x+\mathbf b_1
\rightarrow W_2(\cdot)+\mathbf b_2.
$$

It collapses to one affine transformation.

### Network B — affine + activation

$$
\mathbf x\rightarrow W_1\mathbf x+\mathbf b_1
\rightarrow f(\cdot)
\rightarrow W_2(\cdot)+\mathbf b_2.
$$

The activation interrupts the algebraic collapse.

This is why the usual dense-network pattern is

$$
\boxed{\text{linear/affine}\rightarrow\text{nonlinearity}\rightarrow\text{linear/affine}\rightarrow\text{nonlinearity}\rightarrow\cdots}
$$

---

## 10. Activation Is Not Learning

The activation function usually does not learn its own shape.

For example,

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

is fixed.

Learning changes the weights and biases around it.

So keep the jobs separate:

| Part | Job |
|---|---|
| weights | decide how inputs are combined |
| bias | shift the pre-activation |
| activation | add nonlinearity |
| loss | measure error |
| gradient | tell parameters which direction reduces loss |
| optimizer | choose the update step |

This separation becomes crucial when we study training.

---

## 11. The Derivative Gives Us Another Clue

Gradient-based learning cares about how outputs change when inputs or parameters change.

For ReLU,

$$
\operatorname{ReLU}'(z)=
\begin{cases}
0,&z<0\\
1,&z>0
\end{cases}
$$

At exactly $z=0$, the ordinary derivative is not defined, but practical implementations choose a convention for the backward computation.

For sigmoid,

$$
\sigma'(z)=\sigma(z)(1-\sigma(z)).
$$

At large positive or negative $z$, sigmoid becomes nearly flat, so its derivative becomes very small. This is one reason saturation matters during optimization.

---

## 12. NumPy: Write ReLU Yourself

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

Writing the function yourself makes the concept visible before a framework hides it behind an API.

---

## 13. PyTorch Verification

```python
import torch

z = torch.tensor([-2., -1., 0., 1., 2.])
print(torch.relu(z))
```

The arithmetic is the same. PyTorch additionally connects the operation to automatic differentiation and model training.

---

## 🎮 Activation Playground

Plot ReLU, sigmoid, tanh, Leaky ReLU, GELU, and SiLU on the same horizontal range.

For each function ask:

1. What is the output range?
2. Is it smooth?
3. What happens for a very large positive input?
4. What happens for a very large negative input?
5. Where is the function flat?
6. What might its derivative look like?

Predict first, then inspect the graph.

---

## ⚠️ Failure Mode: Dying ReLU

For an ordinary ReLU neuron, suppose every relevant example produces a negative pre-activation:

$$
z<0.
$$

Then

$$
\operatorname{ReLU}(z)=0
$$

and the derivative through that region is zero.

The neuron can therefore receive no gradient through that path and may remain inactive. This is commonly called the **dying ReLU** problem.

It is one reason alternative activations exist.

---

## 🧠 Common Misconceptions

### “More layers automatically make a network nonlinear.”
No. A stack of affine layers is still affine.

### “ReLU makes a network completely nonlinear everywhere.”
ReLU networks are piecewise linear. They are nonlinear globally, even though each region is linear.

### “Sigmoid is always better because it is smooth.”
Smoothness alone does not guarantee good optimization. Saturation can make gradients tiny.

### “The activation function learns the pattern by itself.”
Usually the activation is fixed. Training primarily changes weights and biases.

---

## 🔬 Think Like a Scientist

Take

$$
f(x)=2x+1.
$$

First calculate $f(x)$ for $x=-3,-2,-1,0,1,2$.

Then calculate

$$
g(x)=\operatorname{ReLU}(f(x)).
$$

Answer before running Python:

- Where does the graph become flat?
- At which $x$ does the bend occur?
- What is the slope before the bend?
- What is the slope after the bend?

Then repeat the experiment with

$$
f(x)=2x-3.
$$

Notice how the bias moves the location of the bend.

---

## 🧩 Exercises

### Level 1 — Calculate

1. Compute ReLU for $[-4,-1,0,2,7]$.
2. Show algebraically that $g(f(x))$ is affine when both $f$ and $g$ are affine.

### Level 2 — Explain

3. Why does stacking affine layers fail to create a richer family of functions?
4. Explain why ReLU is piecewise linear.

### Level 3 — Investigate

5. Draw the function $\operatorname{ReLU}(3x-6)$ and locate its bend.
6. Compare sigmoid and tanh at $z=-5,0,5$.

### Level 4 — Research Bridge

7. Why did ReLU become popular despite its non-differentiability at exactly zero?
8. Compare ReLU and GELU in terms of shape and gradient behavior.

---

## 🏁 Mastery Gate

Move on only if you can:

- prove that affine layers compose to another affine layer;
- calculate ReLU by hand;
- draw ReLU, sigmoid, and tanh;
- explain what “piecewise linear” means;
- explain why nonlinearity makes depth useful;
- connect activation shape to derivative behavior;
- describe one activation failure mode.

## What You Should Remember

> **Depth becomes powerful when we insert nonlinearity between affine transformations.**

The basic neuron is

$$
z=\mathbf w^T\mathbf x+b,
$$

and the activated neuron is

$$
\boxed{a=f(\mathbf w^T\mathbf x+b)}.
$$

The activation function is the small piece that prevents the whole network from collapsing into one giant straight-line equation.

**Next:** we use many nonlinear neurons together and see how they can approximate complicated functions.
