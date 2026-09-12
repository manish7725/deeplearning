# Lecture 14 — Logistic Regression: The One-Neuron Network

> **The Big Question:** How can a linear model make a yes/no decision without pretending that every output is a real-valued quantity?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2014%20-%20Logistic%20Regression%3A%20The%20One-Neuron%20Network/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Linear regression estimated a real-valued target by combining features linearly and fitting parameters under a Gaussian-noise model.

**Today:** We keep the linear score but change what it means. A sigmoid turns the score into a number between 0 and 1, and the Bernoulli likelihood from Chapter 12 gives the objective.

**Next:** One yes/no output is useful. Real classifiers often choose among many classes, forcing us toward softmax and cross-entropy.

---

## 1. The Problem: Spam or Not Spam?

Suppose an email has two simple features:

- $x_1$: number of suspicious words
- $x_2$: number of links

We want the answer:

```text
spam = 1
not spam = 0
```

Consider:

| Email | suspicious words | links | spam? |
|---|---:|---:|---:|
| A | 0 | 1 | 0 |
| B | 1 | 1 | 0 |
| C | 3 | 2 | 1 |
| D | 4 | 3 | 1 |

A linear model can create a score

$$
z=w_1x_1+w_2x_2+b.
$$

But $z$ can be $-20$, $0$, or $17$.

That is not a probability.

---

## 2. What Would a Solution Need?

We need a model that:

1. can still combine features with a weighted sum,
2. produces a number between 0 and 1,
3. behaves smoothly so calculus still works,
4. can be trained using the observed 0/1 labels,
5. has an objective that comes from a probability model rather than a random score.

The first requirement says **keep the linear part**.

The second forces us to add something after it.

---

## 3. First Attempt: Threshold the Linear Score

Maybe compute $z$ and simply say:

$$
\text{spam}=\begin{cases}
1 & z>0\\
0 & z\le0
\end{cases}
$$

This produces a valid decision.

But what if we want probabilities?

A score of $0.1$ and a score of $10$ both become “spam.” The threshold throws away how strongly the model feels.

Worse, the threshold has a sharp jump:

```text
probability
1 ┤        ┌────────
  │        │
0 ┼────────┘
  └──────────────── score
```

A tiny change around zero can flip the answer completely.

> ⚠️ **A Tempting Wrong Idea**
>
> *“A classifier only needs a threshold, so let’s train the hard yes/no decision directly.”*
>
> A hard threshold is useful at prediction time, but it is awkward for gradient-based training because the function is flat almost everywhere and discontinuous at the threshold.

We need a **soft** version.

---

## 4. The Discovery: The Sigmoid

We want a function that takes any real number and maps it into $(0,1)$.

One particularly useful choice is

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}.
$$

This is the **sigmoid** function.

| Level | The same idea |
|---|---|
| 💡 **Intuition** | A volume knob that never goes below 0% or above 100%. Negative scores lean toward 0; positive scores lean toward 1. |
| ✏️ **Numbers** | $\sigma(0)=0.5$, $\sigma(2)\approx0.8808$, $\sigma(-2)\approx0.1192$. |
| 🎓 **Abstraction** | $p=\sigma(z)$ with $z=\mathbf w\cdot\mathbf x+b$, so $p\in(0,1)$. |

Now our classifier is

$$
\boxed{
 p(y=1\mid\mathbf x)=\sigma(\mathbf w\cdot\mathbf x+b)
}
$$

The linear score has not disappeared.

We simply changed its interpretation:

```text
features
   ↓
weighted sum
   ↓
   z
   ↓
sigmoid
   ↓
probability
```

---

## 5. Why Does the Sigmoid Have This Shape?

The sigmoid is not the only function mapping real numbers to $(0,1)$. It became central here because it interacts beautifully with the odds of a Bernoulli probability.

Start with

$$
 p=\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Then

$$
\frac{p}{1-p}=e^z.
$$

Take logs:

$$
\log\frac{p}{1-p}=z.
$$

So the **log-odds** are linear:

$$
\boxed{\log\frac{p}{1-p}=\mathbf w\cdot\mathbf x+b}.
$$

This is the defining idea behind logistic regression.

The model is not saying “probability is linear.”

It is saying:

> **log-odds are linear in the features.**

---

## 6. Bernoulli Likelihood Returns

Our labels are 0 or 1.

That is exactly the Bernoulli setting from Chapter 12.

For one example:

$$
P(y\mid p)=p^y(1-p)^{1-y}.
$$

Substitute

$$
p=\sigma(\mathbf w\cdot\mathbf x+b).
$$

For $n$ independent examples:

$$
L(\mathbf w,b)
=
\prod_{i=1}^{n}p_i^{y_i}(1-p_i)^{1-y_i}.
$$

Take logs:

$$
\ell(\mathbf w,b)
=
\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right].
$$

And minimize the negative:

$$
\boxed{
J(\mathbf w,b)
=
-\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
}
$$

This is **binary cross-entropy**, also called the Bernoulli negative log-likelihood.

The objective was not invented separately. Chapter 12 already gave it to us.

---

## 7. The Beautiful Gradient Cancellation

This is where logistic regression becomes unusually elegant.

For one example,

$$
z=\mathbf w\cdot\mathbf x+b,
\qquad
p=\sigma(z).
$$

The loss is

$$
\ell=-[y\log p+(1-y)\log(1-p)].
$$

First differentiate the loss with respect to $z$.

The sigmoid derivative is

$$
\sigma'(z)=\sigma(z)(1-\sigma(z))=p(1-p).
$$

Applying the chain rule gives

$$
\frac{\partial\ell}{\partial z}=p-y.
$$

That cancellation is the key result.

Then because

$$
z=\mathbf w\cdot\mathbf x+b,
$$

we get

$$
\boxed{
\frac{\partial\ell}{\partial\mathbf w}=(p-y)\mathbf x
}
$$

and

$$
\boxed{
\frac{\partial\ell}{\partial b}=p-y.
}
$$

For a dataset, sum or average these gradients across examples.

The error term is simply:

```text
prediction − truth
       ↓
      p − y
       ↓
 multiply by input
       ↓
 weight gradient
```

That is one of the places where the architecture of the model and the choice of likelihood fit together almost perfectly.

---

## 8. One Example by Hand

Suppose

$$
\mathbf x=
\begin{bmatrix}
2\\1
\end{bmatrix},
\quad
\mathbf w=
\begin{bmatrix}
1\\-0.5
\end{bmatrix},
\quad
b=-0.5,
\quad y=1.
$$

First calculate the linear score:

$$
z=1(2)+(-0.5)(1)-0.5=1.
$$

Then

$$
p=\sigma(1)=\frac1{1+e^{-1}}\approx0.7311.
$$

The prediction is below the truth, so

$$
p-y\approx-0.2689.
$$

Weight gradient:

$$
\frac{\partial\ell}{\partial\mathbf w}
=(-0.2689)
\begin{bmatrix}2\\1\end{bmatrix}
=
\begin{bmatrix}-0.5378\\-0.2689\end{bmatrix}.
$$

Bias gradient:

$$
\frac{\partial\ell}{\partial b}=-0.2689.
$$

A gradient-descent update therefore increases both weights and the bias for this positive example.

---

## 9. The Decision Boundary

The classifier outputs probabilities, but at prediction time we often choose a class using a threshold such as $0.5$.

When does

$$
p=0.5?
$$

For the sigmoid this happens when

$$
z=0.
$$

Therefore the decision boundary is

$$
\boxed{\mathbf w\cdot\mathbf x+b=0.}
$$

With two features, it is a line.

With three features, a plane.

With $d$ features, a $(d-1)$-dimensional hyperplane.

This is why logistic regression produces a **linear decision boundary** even though the output probability is nonlinear in $z$.

---

## 10. Probability and Class Are Different Objects

This distinction matters.

A model can say

$$
p=0.51
$$

and a threshold rule might call that class 1.

But the model did **not** say “I am 100% sure.” It said the estimated probability of class 1 is 51% under the model.

The threshold is a decision rule placed on top of the probability model.

> 🧠 **Think** — What happens if the cost of a false negative is much larger than the cost of a false positive? A $0.5$ threshold may no longer be the best decision rule, even though the probability model stays the same.

---

## 🔬 The Experiment

In the notebook:

1. Plot the sigmoid from $-8$ to $8$.
2. Predict the probability for a few hand-picked scores.
3. Train logistic regression from scratch on the spam-style dataset.
4. Visualize the decision boundary.
5. Move one example and predict how the boundary will move before running the experiment.

---

## How It Breaks

| Failure | Symptom | Why |
|---|---|---|
| Treat a raw score as a probability | values below 0 or above 1 | linear scores are unconstrained |
| Train a hard threshold | gradient gives little useful information | discontinuity / flat regions |
| Forget the bias | boundary is forced through the origin | model cannot translate the boundary |
| Interpret 0.8 as certainty | probability treated as truth | it is model-based uncertainty, not a guarantee |
| Use squared error automatically | poor classification objective | it does not match the Bernoulli likelihood as directly as log loss |
| Confuse probability with decision | threshold changes model output | threshold is a separate decision rule |

---

## Shapes

For $n$ examples and $d$ features:

```text
X           : (n, d)
weights     : (d,)
bias        : scalar
z           : (n,)
p           : (n,)
y           : (n,)
loss        : scalar
```

For one example:

```text
x : (d,)
w : (d,)
w @ x : scalar
```

The scalar $z$ is the bridge from vector features to probability.

---

## 🎯 Machine Learning Connection

Logistic regression is the smallest model that looks like a neural-network neuron:

$$
\boxed{p=\sigma(\mathbf w\cdot\mathbf x+b)}
$$

It already contains the pieces we will see again and again:

- weighted input,
- bias,
- nonlinear activation,
- loss,
- gradient,
- gradient descent.

The next step is not to invent a new architecture. It is to ask what happens when one probability is not enough.

---

## Distinctions That Matter

| Pair | Difference |
|---|---|
| Linear regression vs logistic regression | Real-valued target with Gaussian-style noise vs binary target with Bernoulli likelihood. |
| Score vs probability | $z$ is unconstrained; $p$ lives in $(0,1)$. |
| Sigmoid vs threshold | Sigmoid produces probability; threshold turns probability into a class decision. |
| Likelihood vs loss | Likelihood is maximized; negative log-likelihood is minimized. |
| Coefficient vs odds ratio | A coefficient changes log-odds; $e^{w_j}$ is the multiplicative change in odds for a one-unit feature increase, holding others fixed. |

---

## What We Discovered

1. Binary classification needs a probability between 0 and 1.
2. A linear score can be turned into a probability with the sigmoid.
3. The sigmoid makes log-odds linear in the features.
4. The Bernoulli likelihood gives binary cross-entropy as negative log-likelihood.
5. The gradient simplifies to $(p-y)x$ for the weights.
6. The 0.5 threshold creates a linear decision boundary because it corresponds to $z=0$.

---

## Mathematics We Built

$$
z=\mathbf w\cdot\mathbf x+b
$$

$$
\sigma(z)=\frac1{1+e^{-z}}
$$

$$
p(y=1\mid\mathbf x)=\sigma(z)
$$

$$
\log\frac{p}{1-p}=z
$$

$$
J=-\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]
$$

$$
\frac{\partial\ell}{\partial z}=p-y
$$

$$
\nabla_{\mathbf w}\ell=(p-y)\mathbf x
$$

$$
\frac{\partial\ell}{\partial b}=p-y
$$

$$
\mathbf w\cdot\mathbf x+b=0
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $z$ | “z” | linear score | `z` |
| $\sigma$ | “sigma” | sigmoid function | `sigmoid` |
| $p$ | “probability” | predicted probability of class 1 | `p` |
| $y$ | “label” | observed 0/1 class | `y` |
| $\mathbf w$ | “w vector” | feature weights | `weights` |
| $b$ | “bias” | intercept | `bias` |
| $J$ | “loss” | total binary cross-entropy | `loss` |

---

## One-Minute Explanation

Start with the same weighted sum used in linear regression. Call its result a score. The problem is that a score can be any real number, while a probability must be between 0 and 1. The sigmoid converts the score into a probability. Because the labels are 0 or 1, the Bernoulli likelihood gives the correct probabilistic objective, whose negative log is binary cross-entropy. Gradient descent then changes the weights so predicted probabilities become more consistent with observed labels. A 0.5 threshold is only the final decision rule, not the probability model itself.

---

## Exercises

### Level 1 — Observe
Read a sigmoid graph. Which input values correspond to probabilities below 0.1 and above 0.9?

### Level 2 — Calculate
Compute $\sigma(0)$, $\sigma(1)$ and $\sigma(-1)$ approximately.

### Level 3 — Derive
Starting from $p=1/(1+e^{-z})$, derive $\log(p/(1-p))=z$.

### Level 4 — Investigate
Train the classifier, move one positive example farther from the boundary, and predict how the fitted weights will change.

### Level 5 — Design
Suppose false negatives cost ten times as much as false positives. Design a decision rule that uses the same probability model but a different threshold. Explain why training and decision-making are separate steps.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Calling sigmoid itself the loss | Sigmoid is the output transformation; binary cross-entropy is the objective. |
| Saying logistic regression is “nonlinear regression” | The probability as a function of features is nonlinear, but the log-odds are linear. |
| Treating 0.5 as a law | It is a common threshold, not a universal decision-theoretic requirement. |
| Forgetting numerical stability | Computing `log(sigmoid(z))` naively can underflow for large magnitude scores; stable implementations use equivalent identities. |

---

## Socratic Questions

- Why is a linear score not already a probability?
- Why does the 0.5 probability correspond exactly to $z=0$?
- Why does Bernoulli likelihood produce a different loss from Gaussian likelihood?
- What would break if we replaced the sigmoid with a hard threshold during training?
- Why can the same probability model support different decision thresholds?

---

## 🔭 Bridge to Chapter 15

One classifier gives us:

```text
p(class 1 | x)
```

But suppose a photograph could be **cat, dog, horse, bird, or car**.

We need probabilities for **many classes** that add up to 1.

> **How can one linear score per class become a single probability distribution?**

That question leads to the **softmax function**, and the same maximum-likelihood logic leads to multiclass cross-entropy.