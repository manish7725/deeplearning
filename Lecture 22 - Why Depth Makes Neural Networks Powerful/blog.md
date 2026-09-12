# Lecture 22 — Why Depth Makes Neural Networks Powerful

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/main/Lecture%2022%20-%20Why%20Depth%20Makes%20Neural%20Networks%20Powerful/notebook.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2022%20-%20Why%20Depth%20Makes%20Neural%20Networks%20Powerful/notebook.ipynb)**

The blog is the textbook; the notebook is the laboratory. Predict the shape of the representation before you run the experiment.

## 🧭 Where this lesson fits

**Previous:** Lecture 21 — Activation Functions: Bending the Straight Line.

**Today:** We stack nonlinear layers and discover what depth actually buys us.

**Next:** Lecture 23 — Backpropagation: Sending the Error Backward.

The key bridge is:

> **Depth lets a network build complicated functions from repeated simple transformations.**

---

## 1. One layer sees the world directly

A single neuron computes

$$
a=f(wx+b).
$$

A layer computes

$$
\mathbf a=f(W\mathbf x+\mathbf b).
$$

It takes the original input and creates a new representation.

Think of this as translating the problem into a new coordinate system that may make the final task easier.

---

## 2. Two layers: transform, then transform again

Now use two layers:

$$
\mathbf h=f(W_1\mathbf x+\mathbf b_1)
$$

and

$$
\mathbf a=f(W_2\mathbf h+\mathbf b_2).
$$

Read this literally:

```text
input
  ↓
weighted sum
  ↓
activation
  ↓
hidden representation
  ↓
weighted sum
  ↓
activation
  ↓
output representation
```

The second layer does not see the raw input alone.

It sees what the first layer has already discovered.

That is the beginning of **hierarchical representation learning**.

---

## 3. A simple example: building a complicated shape

Suppose a one-dimensional input $x$ goes through a few ReLU units.

Each ReLU unit creates a function that is flat on one side and linear on the other:

$$
\max(0,x-c).
$$

By combining several shifted ReLUs, we can build a piecewise-linear function.

For example:

$$
f(x)=\max(0,x-1)-\max(0,x-3).
$$

Let's inspect it:

- for $x<1$, both terms are zero;
- between $1$ and $3$, the first term grows while the second is zero;
- after $3$, both grow and partially cancel.

The result is not one straight line.

It has **pieces**.

This is the first important geometric idea behind ReLU networks.

---

## 4. A network builds pieces on top of pieces

A shallow network can create many linear regions.

A deeper network can transform the output of those regions and create still more complicated partitions.

You can imagine:

```text
Layer 1:  ─╲__╱──╲__╱─
             ↓
Layer 2:   ╲____╱╲___╱
               ↓
Layer 3:     ╲__╱╲__╱╲__
```

The drawing is only intuition. The actual computation is matrix multiplication and nonlinear activation.

The important concept is **composition**.

---

## 5. Composition is the mathematical heart of depth

Suppose

$$
f_1(\mathbf x)=\sigma(W_1\mathbf x+\mathbf b_1)
$$

and

$$
f_2(\mathbf h)=\sigma(W_2\mathbf h+\mathbf b_2).
$$

Then the network computes

$$
\mathbf a=f_2(f_1(\mathbf x)).
$$

A deeper network simply composes more functions:

$$
F(\mathbf x)=f_L(f_{L-1}(\cdots f_2(f_1(\mathbf x))\cdots)).
$$

This is why the word **deep** matters: there are many transformations between input and output.

---

## 6. Why is a hidden representation useful?

Imagine predicting whether a photograph contains a cat.

The first layer does not need to say “cat.”

It might respond to local patterns such as:

```text
edges
corners
simple textures
```

A later layer can combine those into:

```text
ear-like shape
fur-like texture
eye-like pattern
```

And a later representation can combine those into:

```text
cat-like object
```

This is an intuition, not a claim that every network learns exactly these human-named features. The deeper lesson is that successive layers can construct increasingly task-useful representations.

---

## 7. The same idea appears outside images

Depth is not only about pictures.

For text, successive transformations can represent increasingly contextual information.

For audio, early layers can capture local temporal patterns while later layers combine them into richer structures.

For tabular data, deeper transformations can model interactions among features.

The common pattern is:

$$
\text{simple representation}
\rightarrow
\text{richer representation}
\rightarrow
\text{task-relevant representation}.
$$

---

## 8. Width and depth are different

A network can become more powerful in at least two obvious directions:

**Width:** more neurons in a layer.

**Depth:** more layers.

For example:

```text
wide and shallow:
input → [many neurons] → output

narrower and deeper:
input → [few] → [few] → [few] → output
```

These are not interchangeable in practice.

Changing width changes how many features can be represented at one stage.

Changing depth changes how many transformations can be composed.

---

## 9. More depth also means more parameters

Suppose a dense layer has $d$ inputs and $m$ neurons.

It has

$$
m(d+1)
$$

parameters.

A deep network adds these counts layer by layer.

For example:

```text
10 inputs → 8 hidden → 4 hidden → 1 output
```

Parameter counts:

$$
8(10+1)=88
$$

$$
4(8+1)=36
$$

$$
1(4+1)=5
$$

Total:

$$
88+36+5=129.
$$

Depth is not free. More computation and memory are required.

---

## 10. A deeper network is a longer computation graph

Write the forward pass as:

$$
\mathbf x
\rightarrow z_1
\rightarrow a_1
\rightarrow z_2
\rightarrow a_2
\rightarrow z_3
\rightarrow a_3.
$$

The network is now a chain of dependencies.

That chain has a consequence.

If the final answer is wrong, how do we know which earlier parameter should change, and by how much?

This is exactly the question backpropagation answers.

---

## 11. A tiny numerical network

Take a scalar input and two layers:

$$
h=\operatorname{ReLU}(2x+1)
$$

$$
\hat y=3h-4.
$$

For $x=2$:

$$
h=\operatorname{ReLU}(5)=5
$$

and

$$
\hat y=3(5)-4=11.
$$

Now change the input to $x=-1$:

$$
h=\operatorname{ReLU}(-1)=0
$$

so

$$
\hat y=-4.
$$

The first activation changes the region of the function that the second layer sees.

---

## 12. ReLU creates piecewise-linear networks

This fact is worth understanding carefully.

A ReLU network is made from affine operations plus ReLU.

Each individual ReLU is piecewise linear.

Compositions of these operations therefore produce functions that can also be understood as combinations of many linear pieces separated by boundaries.

This gives us a powerful mental picture:

> **A ReLU network is not one giant mysterious formula. It is a machine that assembles many simple linear pieces into a complicated overall shape.**

---

## 13. Why not just use one giant lookup table?

A lookup table could memorize a finite training set.

But a network learns a function that can produce outputs for inputs it has never seen.

The network is useful because it captures reusable transformations instead of storing one answer per example.

This connects directly to the earlier lesson on generalization.

---

## 14. Universal approximation is not the same as easy learning

You may hear the phrase **universal approximation theorem**.

At a high level, certain neural-network architectures can approximate broad classes of functions arbitrarily well under appropriate assumptions.

But do not turn that statement into:

> “A neural network can automatically learn anything.”

The theorem is about representational capacity, not a guarantee that optimization, data, compute, or generalization will be good.

That distinction matters.

---

## 🎮 Interactive depth playground

Build a tiny network with:

```text
x → Linear → ReLU → Linear → ReLU → output
```

Then change one weight at a time.

Watch:

1. the hidden representation;
2. the final prediction;
3. the shape of the function over many input values.

A particularly useful experiment is to compare:

```text
one affine layer
```

against

```text
affine → ReLU → affine
```

and then add another ReLU layer.

Ask yourself:

> **Where did the extra bends come from?**

---

## ⚠️ Common misconceptions

### “Deep means intelligent.”
Depth is an architectural property. It does not guarantee useful learning.

### “More layers always improve accuracy.”
No. Optimization, data, regularization, architecture, and compute all matter.

### “The first layer learns concepts and the last layer learns nothing interesting.”
Real networks are more distributed than that. Representations can be shared and reused across layers.

### “Universal approximation means training is easy.”
No. Expressive capacity and learnability are different questions.

---

## 🔬 Failure mode: too little or too much capacity

A model with too little capacity may underfit.

A model with too much effective capacity may overfit or become difficult to optimize.

That means architecture design is a balance:

$$
\text{capacity} \quad vs. \quad \text{data} \quad vs. \quad \text{optimization}.
$$

---

## 🧩 Exercises

**Level 1 — Calculate**

1. Compute the output of $h=\operatorname{ReLU}(3x-2)$ for $x=-1,0,1,2$.
2. Compute the output of $\hat y=2h+1$ for each value.

**Level 2 — Explain**

3. Why does a second nonlinear layer see a different problem from the original input?
4. Explain the difference between width and depth.

**Level 3 — Investigate**

5. Build a one-hidden-layer ReLU network and plot its output over $x\in[-5,5]$.
6. Add hidden units and observe how many bends the function can create.

**Level 4 — Research bridge**

7. Study the universal approximation theorem and identify exactly what it guarantees—and what it does not.
8. Compare parameter count and compute for shallow versus deep networks with similar width.

---

## 🏁 Mastery gate

Move to Lecture 23 only if you can:

- explain composition of neural-network layers;
- distinguish width from depth;
- explain why ReLU networks create piecewise-linear functions;
- calculate a small two-layer forward pass;
- count parameters across multiple dense layers;
- explain why representational power is not the same as successful training;
- describe a hidden layer as a learned representation.

---

## What you should remember

> **Depth lets simple transformations become a hierarchy of representations.**

The essential mathematical pattern is

$$
\boxed{\mathbf a_L=f_L(W_L f_{L-1}(\cdots f_1(W_1\mathbf x+\mathbf b_1)\cdots)+\mathbf b_L)}.
$$

Do not memorize the giant expression. Read it as:

```text
transform → bend → transform → bend → ...
```

Now there is a serious problem left.

We can compute the output.

We can measure the error.

But **how does the error tell every earlier weight what to change?**

**Next:** backpropagation.
