# Lecture 19 — Support Vector Machines and Kernels

> **The Big Question:** If many boundaries classify the data correctly, why not choose the one that leaves the largest safety margin?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2019%20-%20Support%20Vector%20Machines%20and%20Kernels/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Ensembles combined many trees. Forests reduced instability by averaging diversified models; boosting built a model through successive corrections.
**Today:** We return to geometry. Instead of asking for many boundaries, we ask which single boundary separates two classes with the largest possible margin. Then we discover the kernel trick: work with similarity without explicitly writing every transformed feature.
**Next:** A model can have a beautiful separating boundary and still fail outside its training sample. We need to learn how to tell whether our model really learned.

---

## 1. The Problem: Many Lines Can Separate the Same Data

Consider a tiny two-class dataset:

| Point | $x_1$ | $x_2$ | Class |
|---|---:|---:|---|
| A | 1 | 2 | −1 |
| B | 2 | 1 | −1 |
| C | 4 | 4 | +1 |
| D | 5 | 5 | +1 |

A diagonal line somewhere between the two groups can classify all four points correctly.

But there are many such lines.

```text
x₂
 ↑
5|                 + D
4|             + C
3|
2|  − A
1|      − B
 +----------------------→ x₁
```

One separator might pass very close to A. Another might sit safely in the empty space between the classes.

Both can be correct on the training examples.

So correctness alone does not tell us which boundary is safer.

---

## 2. What Would a Solution Need?

A good separator should:

1. classify the training examples correctly;
2. avoid sitting unnecessarily close to either class;
3. depend on a small number of important boundary examples rather than every point equally;
4. allow nonlinear relationships when a straight line is not enough.

The first three suggest a geometric objective. The fourth will force us to rethink what “linear” means.

---

## 3. First Attempt: Pick Any Correct Boundary

Suppose two lines both classify every training example correctly.

```text
Boundary A: close to the negative class
Boundary B: centered between the classes
```

Why might B be better?

Imagine measuring the shortest distance from the boundary to the nearest training point. If B leaves more empty space on both sides, a small measurement error is less likely to flip the prediction.

> ⚠️ **A Tempting Wrong Idea**
>
> *“Any line that gets every training label right is equally good.”*
>
> The training labels only tell us which side each point belongs to. They do not tell us how fragile the boundary is. Two perfect training classifiers can have very different geometric safety margins.

That margin is the quantity we should optimize.

---

## 4. The Discovery: The Margin

Write a linear decision boundary as

$$
\mathbf{w}^T\mathbf{x}+b=0.
$$

The sign tells us the predicted class:

$$
\hat y = \operatorname{sign}(\mathbf{w}^T\mathbf{x}+b).
$$

The training requirement can be written compactly as

$$
 y_i(\mathbf{w}^T\mathbf{x}_i+b) > 0.
$$

Why multiply by $y_i$?

- If $y_i=+1$, we want the score positive.
- If $y_i=-1$, we want the score negative, and multiplying by −1 makes the requirement positive again.

Now we need the distance from a point to the boundary.

For a plane

$$
\mathbf{w}^T\mathbf{x}+b=0,
$$

the perpendicular distance of $\mathbf{x}_0$ from the boundary is

$$
\frac{|\mathbf{w}^T\mathbf{x}_0+b|}{\|\mathbf{w}\|}.
$$

The denominator appears because scaling $\mathbf{w}$ scales the raw score but does **not** move the geometric boundary.

That is the important invariant.

---

## 5. Why We Need a Convention

Notice something strange.

The equations

$$
\mathbf{w}^T\mathbf{x}+b=0
$$

and

$$
2\mathbf{w}^T\mathbf{x}+2b=0
$$

describe exactly the same boundary.

If we tried to “maximize” the raw score $y_i(\mathbf{w}^T\mathbf{x}_i+b)$, we could simply multiply $\mathbf{w}$ and $b$ by a huge number.

That would make the score arbitrarily large without changing the line at all.

So we fix the scale by choosing the nearest training examples to satisfy

$$
 y_i(\mathbf{w}^T\mathbf{x}_i+b)=1.
$$

These boundary examples are the **support vectors**.

The two margin boundaries are therefore

$$
\mathbf{w}^T\mathbf{x}+b=+1
$$

and

$$
\mathbf{w}^T\mathbf{x}+b=-1.
$$

Their distance is

$$
\frac{2}{\|\mathbf{w}\|}.
$$

So maximizing the margin is the same as minimizing $\|\mathbf{w}\|$.

For convenience we minimize its square:

$$
\boxed{\min_{\mathbf{w},b}\frac12\|\mathbf{w}\|^2}
$$

subject to

$$
\boxed{y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1\quad\text{for every training point}.}
$$

This is the **hard-margin support vector machine** objective.

### Three levels

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Put a road between two groups and make the road as wide as possible. |
| ✏️ **Numbers** | If $\|\mathbf{w}\|=2$, the full margin width is $2/2=1$. If $\|\mathbf{w}\|=4$, it is $2/4=0.5$. Smaller weight norm means a wider road. |
| 🎓 **Abstraction** | Minimize $\frac12\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1$. |

---

## 6. The Support Vectors Do the Important Work

Imagine 100 points, but only 4 lie closest to the separating boundary.

Move one far-away point a little. The optimal separator may not move at all.

Move a support vector enough and the boundary can move.

That is why the name matters:

> **Support vectors are the training points that pin down the maximum-margin boundary.**

They are not “the positive examples” or “the closest points to the origin.” They are the points touching the margin under the chosen optimum.

---

## 7. What if the Data Are Not Perfectly Separable?

Real datasets are messy.

A single mislabeled house may sit inside the wrong class.

Hard-margin SVM says every training point must satisfy

$$
 y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1.
$$

That may be impossible.

So we soften the rule with **slack variables** $\xi_i\ge0$:

$$
 y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1-\xi_i.
$$

Now the objective becomes

$$
\boxed{\min_{\mathbf{w},b,\xi}\frac12\|\mathbf{w}\|^2+C\sum_i\xi_i}
$$

The parameter $C$ controls the trade-off:

- large $C$: strongly punish margin violations;
- smaller $C$: tolerate more violations in exchange for a potentially wider margin.

> 💡 **Think** — This is the same kind of trade-off we met with regularization. We are balancing fit against a simpler, more robust decision boundary.

---

## 8. A Connection to Regularization

The SVM objective contains an explicit preference for small $\|\mathbf{w}\|$.

That is a regularization idea:

```text
fit the data
    +
prefer controlled model complexity
```

This is why SVM belongs naturally in the broader story of machine learning rather than being only a geometric curiosity.

---

## 9. The Problem Reappears: What if a Straight Line Cannot Separate the Classes?

Consider points arranged like this:

```text
           +
      −           −

      −           −
           +
```

A straight line cannot separate the two classes cleanly.

We could invent a new feature such as

$$
\phi(x_1,x_2)=x_1^2+x_2^2.
$$

Now distance from the origin becomes a feature, and perhaps the classes become linearly separable in the transformed space.

This works.

But imagine we need thousands or millions of transformed features.

Explicitly constructing them becomes expensive or impossible.

That forces the next discovery.

---

## 10. The Discovery: Similarity Instead of Explicit Features

Suppose $\phi(\mathbf{x})$ maps the original input into a richer feature space.

A linear model there uses dot products

$$
\phi(\mathbf{x})^T\phi(\mathbf{z}).
$$

Do we actually need to construct $\phi(\mathbf{x})$ and $\phi(\mathbf{z})$ separately?

What if a function could compute their inner product directly?

Define

$$
\boxed{K(\mathbf{x},\mathbf{z})=\phi(\mathbf{x})^T\phi(\mathbf{z})}.
$$

This is a **kernel**.

The remarkable part is that the model can operate using pairwise similarities without explicitly materializing the high-dimensional coordinates.

---

## 11. Tiny Kernel Example

Use a simple two-dimensional feature map:

$$
\phi(x)=
\begin{bmatrix}
 x_1^2\\
 \sqrt2 x_1x_2\\
 x_2^2
\end{bmatrix}.
$$

Take

$$
\mathbf{x}=\begin{bmatrix}1\\2\end{bmatrix},
\qquad
\mathbf{z}=\begin{bmatrix}3\\4\end{bmatrix}.
$$

Then

$$
\phi(\mathbf{x})=
\begin{bmatrix}
1\\2\sqrt2\\4
\end{bmatrix}
$$

and

$$
\phi(\mathbf{z})=
\begin{bmatrix}
9\\12\sqrt2\\16
\end{bmatrix}.
$$

Their dot product is

$$
1(9)+(2\sqrt2)(12\sqrt2)+4(16)
=9+48+64
=121.
$$

Now notice:

$$
(\mathbf{x}^T\mathbf{z})^2=(1\cdot3+2\cdot4)^2=11^2=121.
$$

So we can compute the same similarity as

$$
\boxed{K(\mathbf{x},\mathbf{z})=(\mathbf{x}^T\mathbf{z})^2}
$$

without explicitly constructing the three-dimensional feature vector.

That is the **kernel trick**.

---

## 12. Why This Is Not Magic

The kernel trick does not create information from nowhere.

It computes an inner product in a feature space using a cheaper expression in the original space.

The geometry has changed; the algebra lets us avoid writing the transformed coordinates explicitly.

> 🧠 **The deepest idea:** a nonlinear boundary in the original space can be a linear boundary in a richer feature space.

---

## 📜 History Lens — Vapnik and Cortes

Imagine you are **Vladimir Vapnik** working on statistical learning theory and asking a geometric question: how should a classifier trade fitting the training sample against controlling complexity?

Support-vector methods grew from this margin-based view. The maximum-margin formulation became a practical learning method, and the kernel framework later made nonlinear decision boundaries possible through inner products in transformed spaces.

The historical lesson is worth keeping separate from the vocabulary: **the important shift was from merely fitting labels to controlling geometry and capacity.**

---

## The Geometry

For a two-dimensional linear classifier:

```text
negative class       margin       positive class
  −  −  −              |              +  +  +
      −         ------ | ------         +
              w · x+b=0
```

The vector $\mathbf{w}$ is perpendicular to the decision boundary.

The support vectors touch the two margin lines.

The full margin width is

$$
\frac{2}{\|\mathbf{w}\|}.
$$

So the SVM is literally trying to build the widest possible road between the classes.

---

## 🔬 The Experiment

Change exactly one variable: the SVM regularization parameter $C$.

Predict first:

> If $C$ becomes very large, will the model tolerate training violations more or less? Will the boundary become more willing to chase individual training points?

Then compare several values of $C$ in the notebook.

---

## How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| Hard margin on noisy data | impossible or brittle separator | every point must satisfy the constraint |
| Huge $C$ | boundary chases difficult examples | violations are heavily punished |
| Tiny $C$ | many margin violations | fit is sacrificed for a wider/softer margin |
| Poor feature scaling | one feature dominates geometry | distances and dot products depend on scale |
| Overly flexible kernel | excellent training fit, poor generalization | transformed feature space can be very expressive |

---

## Shapes

For $n$ examples and $d$ features:

```text
X              : (n, d)
y              : (n,)
w              : (d,)
b              : scalar
scores         : (n,)
Gram matrix K  : (n, n)
```

The kernel matrix is especially important:

$$
K_{ij}=K(x_i,x_j).
$$

There is one similarity value for every pair of training examples, so the matrix has shape $(n,n)$.

Shape alone predicts the computational cost: storing all pairwise similarities scales quadratically with the number of examples.

---

## 🎯 Machine Learning Connection

SVMs are a compact meeting point of geometry, optimization and regularization.

The margin gives a geometric reason for controlling model complexity. The soft-margin parameter $C$ expresses a fit-versus-violation trade-off. Kernels let us build nonlinear decision functions from similarity computations.

In practice, SVMs can be strong choices on moderate-sized, well-scaled datasets, especially when the feature dimension is large relative to the number of examples.

---

## Distinctions That Matter

| Confusable pair | Real distinction |
|---|---|
| decision boundary vs margin | boundary classifies; margin measures safety around it |
| support vector vs any training point | support vector is active at the optimum / on the margin (or violating it in soft-margin formulations) |
| hard margin vs soft margin | zero tolerance for violations vs explicit slack |
| linear SVM vs kernel SVM | linear separator in input space vs linear separator in an implicit feature space |
| kernel vs feature map | kernel computes inner products induced by a feature map; it need not expose the coordinates |

---

## What We Discovered

1. Many separating boundaries can fit the same data.
2. The margin gives us a geometric preference for a safer separator.
3. The SVM fixes the scale ambiguity of $\mathbf{w}$ and $b$ through a normalization convention.
4. Support vectors are the points that determine the margin.
5. Slack variables make the method usable when data are noisy or nonseparable.
6. Kernels let us use richer feature spaces without explicitly constructing every feature coordinate.
7. A nonlinear boundary in the original space can be linear in a transformed space.

---

## Mathematics We Built

Decision function:

$$
f(x)=\mathbf{w}^T\mathbf{x}+b
$$

Hard-margin constraints:

$$
y_i f(x_i)\ge1
$$

Margin width:

$$
\frac{2}{\|\mathbf{w}\|}
$$

Hard-margin objective:

$$
\min \frac12\|\mathbf{w}\|^2
$$

Soft-margin objective:

$$
\min \frac12\|\mathbf{w}\|^2+C\sum_i\xi_i
$$

Kernel:

$$
K(x,z)=\phi(x)^T\phi(z)
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $\mathbf{w}$ | “w vector” | perpendicular vector controlling the boundary | `coef_` |
| $b$ | “bias” | shifts the boundary | `intercept_` |
| $y_i$ | “y sub i” | class label, usually −1 or +1 | `y[i]` |
| $\xi_i$ | “xi sub i” | margin violation slack | `slack` |
| $C$ | “see” | penalty for violations | `C` |
| $K(x,z)$ | “kernel of x and z” | feature-space inner product / similarity | `kernel` |
| $\phi(x)$ | “phi of x” | transformed feature representation | implicit / `phi_x` |

---

## One-Minute Explanation

A support vector machine does not merely ask for a boundary that works. It asks for a boundary with the **widest possible margin** between the classes.

The closest training examples support the boundary, so they are called support vectors.

When a perfect separator is impossible, slack variables allow some mistakes, controlled by $C$.

When a straight line is not enough, a kernel lets the model behave as if the data had been transformed into a richer feature space — without explicitly writing every transformed coordinate.

---

## Exercises

### Level 1 — Observe
On a diagram, identify the decision boundary and the two margin lines.

### Level 2 — Calculate
If $\|\mathbf{w}\|=5$, what is the full margin width?

### Level 3 — Derive
Starting from the point-to-plane distance formula, derive why the SVM margin is $2/\|\mathbf{w}\|$.

### Level 4 — Investigate
Change only `C` in the notebook. Plot the decision boundary and count training margin violations.

### Level 5 — Design
Design a classifier for concentric circles. Explain why a linear SVM fails and what kind of kernel could make the problem easier.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| “SVM finds any separating line.” | It seeks a maximum-margin separator under its objective. |
| “Larger $\|w\|$ means wider margin.” | The margin is inversely proportional to $\|w\|$. |
| “Support vectors are all training points.” | Only active/critical points determine the optimum geometry. |
| “Kernel means a neural-network feature extractor.” | A kernel defines pairwise inner products; it is a different construction. |
| “Huge $C$ is always better.” | It can force the model to chase noisy or difficult points. |

---

## Socratic Questions

Why must we normalize the SVM constraints before talking about a margin?

Why does scaling every weight and the bias leave the geometric boundary unchanged?

Why can a point far from the boundary matter less than one touching the margin?

What computational price do we pay when using an $n\times n$ kernel matrix?

Why does feature scaling matter before fitting an RBF or polynomial SVM?

---

## 🔭 Bridge to Chapter 20

We have now built classifiers with trees, ensembles, margins and kernels. But a high training score, a wide margin, or a beautiful plot still does not answer the scientific question:

> **Did the model actually learn a pattern that generalizes, or did we accidentally measure our own mistakes?**

Chapter 20 turns evaluation into a first-class part of machine learning: train/validation/test splits, overfitting, leakage, appropriate metrics and the discipline of testing models on data they did not learn from.
