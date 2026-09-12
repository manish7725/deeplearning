# Lecture 26 — Neural Networks as Function Approximators

## 🧪 Interactive Lab

The notebook is the laboratory. The blog is where we reason before we experiment.

## 🧭 Where We Are

**Previously:** We discovered that a stack of affine layers is still affine, and that nonlinear activations let neural networks create richer shapes.

**Today:** We make that idea concrete. A neural network can combine many simple nonlinear pieces to approximate a complicated function.

**Next:** We turn from *what a network can represent* to *how it changes its parameters when it is wrong*.

> **Big Question:** How can many tiny neurons combine to approximate a complicated function?

---

## 1. The Problem: Drawing a Curve With Simple Pieces

Suppose the world gives us

$$y=\sin(x).$$

Imagine that we do not know the formula. We only observe examples:

| $x$ | $y$ |
|---:|---:|
| $-\pi$ | $0$ |
| $-\pi/2$ | $-1$ |
| $0$ | $0$ |
| $\pi/2$ | $1$ |
| $\pi$ | $0$ |

One straight line can match one small part of the curve, but it cannot follow the whole wave. We need a model that can **build bends**.

---

## 2. What Would a Good Approximation Need?

A useful function approximator should:

1. match simple patterns closely;
2. create multiple changes in slope or curvature;
3. improve when useful model capacity is added;
4. work on new inputs, not only observed examples;
5. contain parameters that can be learned from data.

Chapter 25 gave us the ingredient for bends: **nonlinearity**.

---

## 3. First Attempt: One Affine Neuron

Start with

$$\hat y=wx+b.$$

No matter how we choose $w$ and $b$, the result is one straight line.

> **One affine neuron gives one global linear rule plus an offset.**

Changing $w$ changes the tilt. Changing $b$ moves the line. Neither creates a bend.

---

## 4. Second Attempt: Add Polynomial Terms

A polynomial is more flexible:

$$\hat y=a_0+a_1x+a_2x^2+a_3x^3+\cdots.$$

This shows the general idea of function approximation: choose a family of functions rich enough to describe the pattern.

But high-degree polynomials can behave badly outside the observed range. Neural networks offer another strategy: combine many learned nonlinear pieces.

---

## 5. The Discovery: A Hidden Layer Builds Pieces

Consider a one-hidden-layer scalar network:

$$\boxed{\hat y=\sum_{j=1}^{m}a_j\,\sigma(w_jx+b_j)+c}.$$

Read it as a recipe:

1. take $x$;
2. send it through several hidden neurons;
3. each neuron computes $w_jx+b_j$;
4. apply activation $\sigma$;
5. weight each hidden output by $a_j$;
6. add them;
7. add the final bias $c$.

The network is a **weighted combination of learned building blocks**.

---

## 6. Understand One Hidden Neuron First

Take ReLU:

$$\sigma(t)=\max(0,t).$$

Consider

$$h(x)=\operatorname{ReLU}(2x-1).$$

The inside becomes zero at

$$2x-1=0\Rightarrow x=\frac12.$$

Therefore

$$
h(x)=
\begin{cases}
0,&x<0.5\\
2x-1,&x\ge0.5.
\end{cases}
$$

One neuron creates a simple piecewise-linear shape with one bend.

---

## 7. Move the Bend

Change the bias:

$$\operatorname{ReLU}(2x-3).$$

The switch moves to $x=1.5$.

Change the weight:

$$\operatorname{ReLU}(4x-3).$$

The switch is at $x=0.75$, with a steeper active slope.

| Parameter | Effect for $\operatorname{ReLU}(wx+b)$ |
|---|---|
| $w$ | controls direction and slope of the active region |
| $b$ | controls where the activation turns on |

The parameters move and reshape the building blocks.

---

## 8. Add Several Hidden Neurons

Consider

$$h_1(x)=\operatorname{ReLU}(x+1),$$

$$h_2(x)=\operatorname{ReLU}(x),$$

$$h_3(x)=\operatorname{ReLU}(x-1).$$

Combine them:

$$\hat y=0.5h_1(x)-1.0h_2(x)+0.5h_3(x).$$

Each hidden unit contributes a simple shape. The output weights decide how strongly those shapes are added or subtracted.

> **A complicated function can be assembled from many simpler functions.**

---

## 9. Work Through a Tiny Example

Take $x=2$.

$$h_1(2)=3,\qquad h_2(2)=2,\qquad h_3(2)=1.$$

Then

$$\hat y=0.5(3)-1(2)+0.5(1)=0.$$

Nothing mysterious happened. It is repeated multiply, add, and ReLU operations.

---

## 10. Why This Can Approximate Complicated Functions

A ReLU unit can create a bend at a chosen location. Several units can create many bends. A linear output layer can combine these pieces into a complicated curve.

```text
simple hinge 1 ──┐
simple hinge 2 ──┼── weighted sum ── complicated shape
simple hinge 3 ──┘
```

This geometric idea is connected to **universal approximation results**: under suitable assumptions, sufficiently wide networks with suitable nonlinear activations can approximate broad classes of functions on compact domains.

Be precise:

> “Can approximate” is a statement about representation capacity, not a guarantee that training, data, or generalization will succeed.

---

## 11. Representation vs Learning

### Representation question

> Is the target function inside, or close to, the family of functions the architecture can express?

### Learning question

> Can the optimization procedure find useful parameters?

### Generalization question

> Does the learned function work on new data?

These can fail independently.

| Question | Failure example |
|---|---|
| Representation | model family is too simple |
| Optimization | updates fail to find a good solution |
| Generalization | training fit is good but new-data performance is poor |

A powerful model can still be difficult to train or easy to overfit.

---

## 12. Underfitting and Overfitting

With too few hidden units, the model may not have enough pieces. That is **underfitting**.

With very high capacity and noisy data, the model may reproduce accidental wiggles. That is **overfitting**.

The goal is not “as big as possible.” It is to capture useful structure and generalize.

---

## 13. Why ReLU Networks Are Piecewise Linear

A ReLU unit is piecewise linear. A finite sum of ReLU units is therefore also piecewise linear.

For one-dimensional input, adding more hidden units can create more breakpoints.

```text
few neurons:      __/¯¯
more neurons:   _/¯\__/¯\_
many neurons: _/¯\__/¯\_/¯\__
```

> **More units can provide more pieces with which to construct the target function.**

More capacity is not automatically better because it can also make fitting noise easier.

---

## 14. NumPy Experiment

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 200)
y = np.sin(2 * x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Target function: sin(2x)")
plt.show()
```

Now compare a low-degree polynomial with a small nonlinear network.

The important question is:

> **How does increasing the number of useful nonlinear pieces change approximation error?**

---

## 15. PyTorch Model Skeleton

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
```

Read it as

$$1\text{ input}\rightarrow16\text{ features}\rightarrow16\text{ features}\rightarrow1\text{ output}.$$

The architecture is simply a composition of operations we already learned.

---

## 🎮 Function Approximation Playground

Change exactly one thing at a time:

1. 1 hidden neuron;
2. 2 hidden neurons;
3. 4 hidden neurons;
4. 16 hidden neurons;
5. ReLU versus tanh;
6. noiseless versus noisy data.

Record:

| Model | Width | Activation | Train error | Validation error | Observation |
|---|---:|---|---:|---:|---|
| baseline | | | | | |
| experiment | | | | | |

Predict the trend before running the experiment.

---

## ⚠️ Failure Mode: Too Much Capacity

A flexible network may learn noise instead of the smooth rule.

You may see:

- very low training error;
- higher validation or test error;
- unnecessary oscillations between observed points.

> **Approximation power answers “what can this model represent?” It does not answer “what should this model learn?”**

Data, regularization, architecture, optimization, and evaluation all matter.

---

## 🧠 Common Misconceptions

### “Universal approximation means one neuron can learn everything.”
No. The result concerns sufficiently expressive networks under mathematical assumptions.

### “If a network can represent a function, training will find it.”
No. Representation and optimization are separate issues.

### “More neurons always improve test accuracy.”
Not necessarily. Capacity can reduce underfitting but also increase overfitting risk.

### “Neural networks approximate functions only by memorizing examples.”
Memorization can happen, but useful function approximation should work on unseen inputs too.

---

## 🔬 Think Like a Scientist

Consider

$$h_1(x)=\operatorname{ReLU}(x),\qquad h_2(x)=\operatorname{ReLU}(x-1).$$

Answer before running Python:

1. Where does each unit turn on?
2. What are their slopes before and after the bend?
3. What does $h_1(x)-h_2(x)$ look like?
4. Could a third ReLU add another bend?

Then plot the result and compare it with your prediction.

---

## 🧩 Exercises

### Level 1 — Calculate

1. Compute $\operatorname{ReLU}(2x-1)$ for $x=-1,0,1,2$.
2. For the three-neuron example in §8, compute the output at $x=-1,0,1,2$.

### Level 2 — Explain

3. Why can several nonlinear functions create a shape that one affine function cannot?
4. Explain the difference between representation capacity and optimization.

### Level 3 — Investigate

5. Build networks with widths 1, 2, 4, 8, and 16 and compare approximation error on $\sin(2x)$.
6. Add noise to the training data. At what capacity does validation error stop improving?

### Level 4 — Research Bridge

7. Read a statement of a universal approximation theorem and identify its assumptions.
8. Compare width and depth as ways of increasing expressive power. What is gained by composition?

---

## 🏁 Mastery Gate

Move on only if you can:

- explain what a function approximator is;
- derive the one-hidden-layer equation;
- calculate a tiny network by hand;
- explain how a hidden ReLU creates a bend;
- distinguish representation, optimization, and generalization;
- explain underfitting and overfitting;
- run an experiment changing network width;
- explain why universal approximation is not a guarantee of successful training.

## What You Should Remember

> **A neural network approximates complicated functions by combining many simple nonlinear pieces whose shapes and strengths are controlled by learned parameters.**

A useful mental model is

$$
\boxed{\text{input}\rightarrow\text{nonlinear features}\rightarrow\text{weighted combination}\rightarrow\text{output}}.
$$

For a one-hidden-layer scalar network:

$$
\boxed{\hat y=\sum_{j=1}^{m}a_j\,\sigma(w_jx+b_j)+c}
$$

The next question is the harder one:

> **How do we change the parameters so that the network actually learns the right function?**

That takes us into gradients and backpropagation.
