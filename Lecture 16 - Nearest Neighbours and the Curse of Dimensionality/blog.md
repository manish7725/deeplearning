# Lecture 16 — Nearest Neighbours and the Curse of Dimensionality

> **The Big Question:** Can we classify a new example without learning a global rule — simply by asking which old examples look most similar?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2016%20-%20Nearest%20Neighbours%20and%20the%20Curse%20of%20Dimensionality/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We just learned **Nearest Neighbours and the Curse of Dimensionality**.

But using it creates a new question.

> **What problem does this idea still leave us unable to solve?**

That question leads naturally to **Decision Trees**.

**Next → Decision Trees.**

## 1. The Problem: What If We Refuse to Learn a Formula?

Suppose we have a tiny housing dataset:

| House | Rooms | Area (hundreds sq ft) | Class |
|---|---:|---:|---|
| A | 2 | 8 | Apartment |
| B | 2 | 10 | Apartment |
| C | 4 | 16 | Villa |
| D | 5 | 18 | Villa |
| E | 3 | 20 | Farmhouse |

A new house arrives:

$$
\mathbf x_* = \begin{bmatrix}3\\11\end{bmatrix}.
$$

We could fit a line. But there is another idea that needs almost no algebra:

> **Look around the new house. What do its neighbours look like?**

The nearest examples are A, B and C. Two say *Apartment*; one says *Villa*. The majority says Apartment.

This is the core idea of **k-nearest neighbours**, or **k-NN**.

### What Would a Solution Need?

A neighbour-based classifier needs to:

1. Define what “near” means.
2. Find the nearest examples quickly enough.
3. Turn their labels into a prediction.
4. Decide how many neighbours to trust.
5. Avoid letting one badly scaled feature dominate distance.

The first requirement sounds easy until we remember that a house has many measurements.

---

## 2. First Attempt: Just Measure Distance

In two dimensions we know the ordinary Euclidean distance:

$$
d(\mathbf x,\mathbf q)=\sqrt{(x_1-q_1)^2+(x_2-q_2)^2}.
$$

For the new house $[3,11]$ and house A $[2,8]$:

$$
d=\sqrt{(3-2)^2+(11-8)^2}
=\sqrt{1+9}
=\sqrt{10}\approx3.16.
$$

For house C $[4,16]$:

$$
d=\sqrt{1+25}=\sqrt{26}\approx5.10.
$$

So A is closer.

### Three Levels

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Find the houses on a map that are physically closest to you. Nearby houses are often similar. |
| ✏️ **Tiny numbers** | Distance from $[3,11]$ to A is $\sqrt{10}\approx3.16$; to C is $\sqrt{26}\approx5.10$. |
| 🎓 **Abstraction** | $d(\mathbf x,\mathbf q)=\|\mathbf x-\mathbf q\|_2=\sqrt{\sum_j(x_j-q_j)^2}$. |

---

## 3. The First Surprise: The Units Can Betray Us

Now add a new feature: **plot area in square feet**.

Suppose our two features are:

- rooms: around 1–6
- square feet: around 500–3000

Take two houses:

$$
A=[2,800],\qquad B=[3,1000].
$$

The raw difference is

$$
[-1,-200].
$$

The square-root distance is dominated by 200. One room hardly matters.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Distance automatically treats all features fairly."*
>
> It does not. Euclidean distance sees numbers, not units. A feature measured on a large numerical scale can drown out everything else.

We therefore often **scale** features before measuring distance.

For example, map each feature to a comparable scale:

$$
\tilde x_j=\frac{x_j-\mu_j}{\sigma_j}
$$

where $\mu_j$ is the feature mean and $\sigma_j$ is its standard deviation.

The important point is not the particular scaling formula. It is the idea: **distance is a modeling choice, and its meaning depends on the coordinate system.**

---

## 4. The Discovery: Let Neighbours Vote

Finding the nearest single example gives **1-NN**.

But one example may be noisy. Suppose the closest house was mislabelled. A safer idea is to ask several neighbours and let them vote.

For $k=3$:

```text
nearest neighbours
A → Apartment
B → Apartment
C → Villa

vote:
Apartment = 2
Villa     = 1

prediction = Apartment
```

The classifier is therefore:

$$
\boxed{\hat y=\operatorname{mode}\{y_i:i\in N_k(x)\}}
$$

where $N_k(x)$ means “the set of the $k$ nearest training examples to $x$.”

A simple change makes the vote care about distance:

$$
\text{vote weight}_i=\frac{1}{d(x,x_i)+\epsilon}.
$$

A very close neighbour gets more say than a distant one.

---

## 5. What Does $k$ Actually Control?

Try the same training set with different $k$.

### $k=1$

The model is extremely flexible. Every training example essentially draws its own tiny territory.

### $k=3$

One unusual point can be outvoted.

### $k=15$

The neighbourhood becomes broad. Local details disappear.

This is the familiar tension:

| Small $k$ | Large $k$ |
|---|---|
| Very local | More global |
| Low bias, high variance | Higher bias, lower variance |
| Sensitive to noise | Smoother decision regions |

The value of $k$ is therefore a model choice, not a law of nature.

---

## 6. The Geometry: Decision Regions Appear From Data Points

Imagine the training examples as colored dots on a plane.

For 1-NN, every point in the plane belongs to its nearest training example. The boundaries between these regions are **Voronoi boundaries**.

```text
      • AAAAAA | BBBBBB •
       AAAAAAA | BBBBBBB
        AAAAAA  |  BBBBB
                |
        CCCCCCCCCCCCC
             • C
```

There is no fitted line hidden underneath. The training points themselves create the regions.

This is the key conceptual contrast with Chapters 13–15:

> **Linear and logistic models learn a global boundary. k-NN stores examples and lets local geometry make the decision.**

---

## 7. The Cost of Refusing to Learn a Model

k-NN sounds wonderfully simple. But simplicity has a price.

Training is almost trivial: store the dataset.

Prediction is expensive: compare the query with many stored examples.

For $N$ training examples, $d$ features, and one query, a brute-force search costs roughly

$$
O(Nd)
$$

per prediction.

That is the opposite of many parametric models, where training can be expensive but prediction is cheap.

For very large datasets, special data structures can accelerate neighbour search in low dimensions, but those structures also face a deeper problem.

---

## 8. The Discovery: High Dimensions Make “Near” Strange

Suppose a feature is uniformly spread between 0 and 1.

In one dimension, a small interval of width $r$ has length

$$
V_1(r)=r.
$$

In two dimensions, a square of side $r$ has area

$$
V_2(r)=r^2.
$$

In ten dimensions,

$$
V_{10}(r)=r^{10}.
$$

This gets tiny incredibly fast.

For $r=0.1$:

$$
0.1^{10}=10^{-10}.
$$

A tiny neighbourhood occupies only one ten-billionth of the unit hypercube.

So if you want enough examples inside that tiny neighbourhood, you need an enormous number of training examples.

That is the **curse of dimensionality**: many algorithms based on local neighbourhoods become data-hungry as dimension grows.

---

## 9. A More Surprising Calculation

Imagine a unit hypercube and ask how wide a centered hypercube must be to contain just 10% of its volume.

If its side length is $r$, then

$$
r^d=0.1.
$$

Therefore

$$
r=0.1^{1/d}.
$$

For $d=2$:

$$
r\approx0.316.
$$

For $d=10$:

$$
r\approx0.794.
$$

For $d=100$:

$$
r\approx0.977.
$$

In 100 dimensions, a cube covering just 10% of the volume already has a side length almost the entire space.

The phrase **“local neighbourhood”** has become much less local.

---

## 10. Why Distances Start Looking Alike

There is another failure.

With many independent coordinates, squared distances add many random contributions:

$$
\|x-q\|^2=\sum_{j=1}^{d}(x_j-q_j)^2.
$$

As $d$ grows, the total is an accumulation of many terms. Relative differences between the nearest and farthest points can shrink.

A useful diagnostic is the **relative contrast**:

$$
R=\frac{d_{\max}-d_{\min}}{d_{\min}}.
$$

When $R$ becomes small, “nearest” is not very different from “far away.” The neighbour signal weakens.

This is why high-dimensional data often needs a better representation before nearest-neighbour methods become useful.

---

## 11. 📜 History Lens — Cover and Hart

Imagine you are Thomas Cover and Peter Hart in the 1960s, asking a bold question: *Can a classifier succeed simply by copying the labels of nearby examples?*

Their work on nearest-neighbour classification helped establish a surprisingly strong theoretical foundation for the method.

The important historical idea is not that nearest neighbours are universally best. It is that **a classifier does not have to begin with a global equation**. Sometimes the geometry of the data itself can be the rule.

Decades later, the same idea would become useful again in recommendation systems, retrieval, metric learning, and embedding spaces — especially after another problem was solved: learning a representation where “near” actually means “similar.”

---

## 12. 🔬 The Experiment

Open the [lab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2016%20-%20Nearest%20Neighbours%20and%20the%20Curse%20of%20Dimensionality/notebook.ipynb).

Predict before running:

1. Which label does 1-NN assign to $[3,11]$?
2. Does scaling the features change the nearest neighbour?
3. Does increasing $k$ make the decision boundary smoother?
4. As dimension increases, does the ratio between nearest and farthest distances get larger or smaller?

---

## 13. How It Breaks

| Failure | What it looks like | Why it happens |
|---|---|---|
| Unscaled features | One feature dominates distance | Units have different numeric ranges |
| $k=1$ with noisy labels | Tiny islands of wrong predictions | Model follows individual examples |
| Very large $k$ | Boundaries become too smooth | Local structure is averaged away |
| Huge dataset | Slow prediction | Many distance calculations |
| Very high dimension | Neighbours are barely distinguishable | Volume and distance concentration problems |
| Irrelevant features | Bad neighbours | Distance spends capacity measuring noise |

---

## 14. Shapes

For $N$ examples and $d$ features:

```text
X_train : (N, d)
y_train : (N,)
x_query  : (d,)
distances: (N,)
neighbors: (k,)
```

Computing all differences for one query can be represented as

$$
X_{train}-x_{query}
$$

with shape $(N,d)$ through broadcasting. Squaring and summing over the feature axis gives $(N,)$ distances.

For a batch of $M$ queries, pairwise distances have shape $(M,N)$.

Shape reasoning immediately tells us why the classifier needs a different operation for a batch: we now need one distance for every **query–training-example pair**.

---

## 15. 🎯 Machine Learning Connection

Nearest-neighbour methods are examples of **non-parametric** or **instance-based** learning. The model does not summarize the entire training set with a small fixed vector of parameters. It keeps the examples and delays the decision until prediction time.

This idea appears far beyond textbook k-NN:

```text
query → find similar items → use their information
```

That same pattern appears in retrieval systems, recommendation, case-based reasoning, and modern embedding search.

The hard part moves from *learning a boundary* to *learning a useful notion of similarity*.

---

## 16. Distinctions That Matter

| Confusable pair | Difference |
|---|---|
| 1-NN vs k-NN | 1-NN trusts one point; k-NN combines several neighbours. |
| Euclidean distance vs similarity | Distance is small for nearby points; similarity is usually designed to be large for related points. |
| Parametric vs instance-based | Parametric models summarize data with learned parameters; instance-based methods retain training examples. |
| Training cost vs prediction cost | k-NN trains cheaply but can predict expensively. |
| High dimension vs many samples | Adding dimensions is not automatically bad; the problem is that sample requirements can grow extremely quickly. |

---

## 17. What We Discovered

1. A classifier can make predictions without learning one global formula.
2. Distance defines what “similar” means.
3. k-NN uses local votes to classify a new point.
4. The choice of feature scaling changes the geometry and therefore the answer.
5. Small $k$ gives flexible, local decisions; large $k$ smooths them.
6. In high dimensions, local volume shrinks relative to the space and distances become less informative.

---

## 18. Mathematics We Built

$$
d(x,q)=\sqrt{\sum_j(x_j-q_j)^2}
$$

$$
\hat y=\operatorname{mode}\{y_i:i\in N_k(x)\}
$$

$$
\tilde x_j=\frac{x_j-\mu_j}{\sigma_j}
$$

$$
V_d(r)=r^d
$$

$$
r=0.1^{1/d}
$$

$$
R=\frac{d_{\max}-d_{\min}}{d_{\min}}
$$

---

## 19. What Each Symbol Means

| Symbol | Meaning | In code |
|---|---|---|
| $x$ | query point | `query` |
| $x_i$ | training point $i$ | `X_train[i]` |
| $d$ | number of features | `n_features` |
| $N$ | number of training examples | `len(X_train)` |
| $k$ | neighbours used for voting | `k` |
| $\mu_j$ | mean of feature $j$ | `mean[j]` |
| $\sigma_j$ | standard deviation of feature $j$ | `std[j]` |
| $N_k(x)$ | nearest-neighbour index set | `neighbor_idx` |

---

## 20. One-Minute Explanation

k-NN does not try to discover one equation for all examples. It stores the training points. For a new point, it measures distance to the stored points, chooses the closest $k$, and lets them vote. This works beautifully when distance really reflects similarity. But distance becomes harder to interpret in many dimensions because the space grows enormously and local neighbourhoods stop being local enough.

---

## 21. Exercises

### Level 1 — Observe
Look at a plotted dataset and identify the likely nearest neighbours of a new point.

### Level 2 — Calculate
Compute the Euclidean distance between $[3,11]$ and $[2,8]$.

### Level 3 — Derive
Starting from a $d$-dimensional cube with side $r$, derive the volume $r^d$.

### Level 4 — Investigate
Run the notebook for dimensions 2, 10, 50, and 100. Plot the distribution of nearest and farthest distances.

### Level 5 — Design
Invent a distance function for houses in which rooms, price, and area have different business importance. Explain why your distance is fairer than raw Euclidean distance.

---

## 22. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Assuming nearest means most similar | The chosen distance may not encode semantic similarity. |
| Using raw features with incompatible units | Distance becomes scale-dependent. |
| Thinking bigger $k$ is always better | Excessive averaging can erase local structure. |
| Thinking k-NN has no hyperparameters | $k$, scaling, distance metric, and weighting all matter. |
| Blaming dimension alone | Irrelevant or redundant features can be more damaging than dimension itself. |

---

## 23. Socratic Questions

Why is 1-NN able to memorize every training point?

Why can scaling one feature change the predicted class?

Why does the volume formula contain an exponent equal to the dimension?

Why does a high-dimensional neighbourhood need so many samples to remain local?

Why might an embedding with 768 dimensions still be useful for nearest-neighbour search?

---

## 🔭 Bridge to Chapter 17

Nearest neighbours gave us a classifier without a global equation, but it has two uncomfortable properties: it stores everything, and its decisions depend on a geometric notion of similarity.

Can we instead build a model that learns a **sequence of simple questions**?

For example:

```text
Is area < 12?
├── yes → Is rooms < 3?
│          ├── yes → Apartment
│          └── no  → Farmhouse
└── no  → Villa
```

That is the beginning of a **decision tree**.