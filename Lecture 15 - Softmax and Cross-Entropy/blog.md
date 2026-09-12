# Lecture 15 — Softmax and Cross-Entropy

> **The Big Question:** How can one model choose between many classes and learn from the difference between what it believed and what was true?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2015%20-%20Softmax%20and%20Cross-Entropy/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Logistic regression learned a binary decision by turning one score into a probability.
**Today:** Our house is no longer simply *cheap* or *expensive*. We want the model to choose among **apartment, villa, and farmhouse**, and we want its learning rule to punish confident wrong answers more than honest uncertainty.
**Next:** A linear model still draws one straight boundary. We need a model that can make decisions without assuming a straight line.

---

## 1. The Problem: One House, Three Possible Labels

Suppose the same house data now describes three kinds of property:

| House | Rooms | Area (hundreds sq ft) | Class |
|---|---:|---:|---|
| A | 2 | 8 | Apartment |
| B | 4 | 16 | Villa |
| C | 3 | 12 | Villa |
| D | 2 | 20 | Farmhouse |

Our logistic model from Chapter 14 knows how to answer one yes/no question. But *apartment*, *villa*, and *farmhouse* are not three separate yes/no problems if we want one coherent answer.

We want the machine to say something like:

```text
Apartment   0.10
Villa       0.75
Farmhouse   0.15
```

Those three numbers have to obey two rules:

1. Every value is between 0 and 1.
2. Together they add to 1.

Then we can read them as probabilities.

### What Would a Solution Need?

The raw model can already produce three **scores**. Call them $z_1,z_2,z_3$:

$$
\mathbf z = \begin{bmatrix}2.0\\5.0\\1.0\end{bmatrix}
$$

But scores are not probabilities. A score can be negative, larger than 1, and the three scores do not have to add to one.

So we need a function that turns arbitrary scores into a probability distribution while preserving an important idea: **a larger score should mean a larger probability**.

---

## 2. First Attempt: Divide by the Total

A smart first idea is obvious:

$$
q_i = \frac{z_i}{z_1+z_2+z_3}
$$

For scores $[2,5,1]$ this gives

$$
\left[\frac{2}{8},\frac{5}{8},\frac{1}{8}\right]
= [0.25,0.625,0.125].
$$

It is tempting because the numbers sum to 1.

But now try perfectly valid model scores with a negative value:

$$
[-2,5,1].
$$

The same rule produces

$$
[-0.5,1.25,0.25],
$$

which is impossible for probabilities.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Just divide every score by the total."*
>
> Normalization makes numbers sum to one, but it does not make them non-negative. More importantly, it is fragile when the total is close to zero.

We need a function that makes every score positive **before** normalizing.

---

## 3. The Discovery: Exponentials Make Scores Positive

The exponential function has a useful property:

$$
 e^z > 0 \quad \text{for every real } z.
$$

So turn each score into a positive number first:

$$
[2,5,1]
\longrightarrow
[e^2,e^5,e^1].
$$

Then divide by their total:

$$
\boxed{p_i = \frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}}
$$

This is **softmax**.

Read the notation slowly:

- $K$ is the number of classes.
- $z_i$ is class $i$'s raw score, called a **logit**.
- $e^{z_i}$ makes the score positive.
- The denominator adds all positive values, so the results sum to 1.

For the tiny example,

$$
\mathbf z=[2,5,1]
$$

we get approximately

$$
\operatorname{softmax}(\mathbf z)
\approx [0.047,0.937,0.017].
$$

The largest score becomes the largest probability, and the gaps between scores control how concentrated the probability is.

---

## 4. Why the Exponential Is Doing Something Interesting

The exponential does not merely make values positive. It changes **differences** into **ratios**.

Suppose two scores differ by 1:

$$
 z_2-z_1=1.
$$

Then

$$
\frac{e^{z_2}}{e^{z_1}} = e^{z_2-z_1}=e.
$$

So a one-unit score advantage means the larger class receives roughly $2.718$ times as much unnormalized weight.

If the gap is 2,

$$
\frac{e^{z_2}}{e^{z_1}}=e^2\approx7.39.
$$

That gives softmax a useful behavior: **small score differences become meaningful probability differences**.

### Three Levels

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Three runners finish close together. Exponential scoring makes a small lead matter more, then normalization converts the scores into shares of the prize. |
| ✏️ **Tiny numbers** | $[2,5,1]\to[e^2,e^5,e^1]\to[7.39,148.41,2.72]\to[0.047,0.937,0.017]$. |
| 🎓 **Abstraction** | $p_i=e^{z_i}/\sum_j e^{z_j}$ gives one probability per class, with $p_i\ge0$ and $\sum_i p_i=1$. |

---

## 5. A Property We Get for Free: Adding the Same Number Changes Nothing

Here is a strange-looking fact:

$$
\operatorname{softmax}([2,5,1])
=
\operatorname{softmax}([12,15,11]).
$$

Why?

Because the same constant $c$ cancels:

$$
\frac{e^{z_i+c}}{\sum_j e^{z_j+c}}
=
\frac{e^c e^{z_i}}{e^c\sum_j e^{z_j}}
=
\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

This tells us that **only score differences matter**.

It also gives us a practical trick. Large logits can overflow numerically, so before exponentiating we subtract the largest score:

$$
\boxed{p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j}
$$

The probabilities are unchanged, but every exponent is now at most $e^0=1$.

That is the stable form used in real software.

---

## 6. Softmax Answers the Many-Class Problem — But How Should It Learn?

Suppose the true class is **Villa**. Our model predicts

$$
\mathbf p=[0.10,0.75,0.15].
$$

We want a loss that says: *How bad was this prediction?*

The answer should have a few sensible properties:

- If the true class gets probability near 1, the loss should be near 0.
- If the true class gets probability near 0, the loss should be very large.
- A confident wrong answer should hurt much more than a mildly wrong answer.
- For three classes, the calculation should still be one clean rule.

---

## 7. First Attempt: Absolute Error on Probabilities

A natural thought is:

$$
L=|y_1-p_1|+|y_2-p_2|+|y_3-p_3|.
$$

For the true one-hot label

$$
\mathbf y=[0,1,0]
$$

and prediction

$$
\mathbf p=[0.10,0.75,0.15],
$$

we get

$$
L=0.10+0.25+0.15=0.50.
$$

But this loss treats a prediction of $0.01$ for the true class as only slightly worse than $0.10$. For learning, that misses something important: **saying "almost impossible" when the truth is "certain" should be dramatically worse.**

We want a loss with a deep penalty near zero probability.

---

## 8. The Discovery: Logarithms Turn Confidence Into Cost

The logarithm has exactly the shape we need:

$$
-\log(1)=0
$$

but

$$
-\log(0.5)\approx0.693,
\qquad
-\log(0.1)\approx2.303,
\qquad
-\log(0.01)\approx4.605.
$$

The closer the true probability gets to zero, the more sharply the loss rises.

For one correct class, the natural loss is therefore

$$
\boxed{L=-\log p_{\text{true}}}.
$$

This is **cross-entropy** for a single example.

If the true class is represented by one-hot vector $\mathbf y$, we can write all classes at once:

$$
\boxed{L=-\sum_{i=1}^{K} y_i\log p_i}
$$

Because exactly one $y_i$ is 1, the sum simply selects the log probability of the true class.

For our example,

$$
L=-\log(0.75)\approx0.288.
$$

---

## 9. Why Softmax and Cross-Entropy Fit Together So Well

Now something beautiful happens. Our model produces logits $z_i$. Softmax turns logits into probabilities. Cross-entropy judges those probabilities.

Substitute softmax into the loss:

$$
L=-\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right)
$$

Use the logarithm rule $\log(a/b)=\log a-\log b$:

$$
L=-\left(z_y-\log\sum_j e^{z_j}\right)
$$

so

$$
\boxed{L=\log\sum_j e^{z_j}-z_y}.
$$

The first term looks like a difficult expression, but it is important enough to have its own name: **log-sum-exp**.

This combined expression explains why libraries often expose one operation called *softmax cross-entropy*: they can compute it stably without explicitly materializing tiny probabilities.

---

## 10. The Gradient Becomes Surprisingly Simple

Here is the key result that makes this pair especially useful for learning:

$$
\boxed{\frac{\partial L}{\partial z_i}=p_i-y_i}
$$

Let us derive it rather than memorize it.

Start with

$$
L=\log\sum_j e^{z_j}-z_y.
$$

For the first term,

$$
\frac{\partial}{\partial z_i}\log\sum_j e^{z_j}
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
=p_i.
$$

For the second term,

$$
\frac{\partial z_y}{\partial z_i}=y_i
$$

when $y_i$ is represented as one-hot data. Therefore,

$$
\frac{\partial L}{\partial z_i}=p_i-y_i.
$$

That is an elegant learning signal:

> **prediction minus truth.**

If the model gives a class too much probability, its logit gets pushed down. If it gives a class too little probability, its logit gets pushed up.

---

## 11. The Running House Example

For a house whose true class is Villa,

$$
\mathbf y=[0,1,0]
$$

and suppose

$$
\mathbf p=[0.10,0.75,0.15].
$$

Then

$$
\frac{\partial L}{\partial \mathbf z}
=
\mathbf p-\mathbf y
=
[0.10,-0.25,0.15].
$$

Read it in plain English:

- Apartment got **too much** probability: push its score down.
- Villa got **too little** probability: push its score up.
- Farmhouse got **too much** probability: push its score down.

The loss is not just a score telling us *how wrong*. Its gradient also tells us *which logits need to move*.

---

## 12. 📜 History Lens — From Entropy to Classification

Imagine you are Claude Shannon in the 1940s, asking a different question: *If an event is uncertain, how many bits does it take to describe its outcome?*

The quantity that emerged was entropy. In machine learning, the same mathematics became a way to measure how surprising an observed class is under a predicted probability distribution.

Cross-entropy is therefore not an arbitrary punishment invented for neural networks. It is tied to the deeper idea that **assigning low probability to what actually happens is costly information-wise**.

The connection matters because it explains the logarithm: a prediction of $0.5$ says the event was plausible; $0.01$ says it was almost ruled out. The logarithm turns those probability ratios into additive costs.

---

## 13. The Geometry

For three classes, the probabilities always satisfy

$$
 p_1+p_2+p_3=1.
$$

So all predictions live on a triangle called the **probability simplex**.

```text
             Villa
               ▲
              / \
             /   \
            /     \
           /       \
          /    •    \
         /           \
        /_____________\
 Apartment           Farmhouse
```

A point near a corner means strong confidence in that class. The center corresponds to roughly equal uncertainty.

Softmax does something important geometrically: every logits vector in $\mathbb R^K$ gets mapped into this probability simplex.

---

## 14. 🔬 The Experiment

Open the [lab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2015%20-%20Softmax%20and%20Cross-Entropy/notebook.ipynb).

Before running it, predict:

1. Which probability is largest for logits `[2, 5, 1]`?
2. Does adding `100` to every logit change the probabilities?
3. Which loss is larger: `-log(0.8)` or `-log(0.2)`?
4. For true class `[0,1,0]`, what is the sign of each gradient component when prediction is `[0.1,0.7,0.2]`?

The notebook computes the answers from scratch and then checks them against the vectorized implementation.

---

## 15. How It Breaks

| Failure | What it looks like | Why it happens |
|---|---|---|
| Divide by raw score | Negative “probabilities” | Scores can be negative |
| Naive `exp` | `inf` / `nan` | Exponential overflow |
| True probability near 0 | Huge loss | The model was confidently wrong |
| Forget normalization | Scores do not sum to 1 | Exponentials alone are not probabilities |
| Treat classes independently | Probabilities need not sum to 1 | Binary sigmoid is not a shared multi-class distribution |

---

## 16. Shapes

For one example with $d$ input features and $K$ classes:

```text
x        : (d,)
W        : (K, d)
b        : (K,)
z        : (K,)
p        : (K,)
y        : (K,)
loss     : scalar
```

The forward pass is

$$
\mathbf z = W\mathbf x+\mathbf b
$$

then

$$
\mathbf p=\operatorname{softmax}(\mathbf z).
$$

For a batch of $N$ examples:

```text
X : (N, d)
W : (K, d)
b : (K,)
Z : (N, K)
P : (N, K)
Y : (N, K)
```

Shape-checking is already enough to reject many incorrect formulas before calculating anything.

---

## 17. 🎯 Machine Learning Connection

Softmax is the standard way to turn multi-class logits into a categorical probability distribution. Cross-entropy then creates a loss that rewards probability mass on the correct class and strongly punishes confident mistakes.

Together they are the mathematical core of ordinary multi-class classification models, from a simple linear classifier to a neural network's final layer.

Notice the division of responsibility:

```text
features → linear scores → softmax probabilities → cross-entropy loss
                         ↑                         ↑
                    "what I believe"          "how costly that belief was"
```

---

## 18. Distinctions That Matter

| Confusable pair | Difference |
|---|---|
| Logit vs probability | A logit is an unconstrained score; a probability is in $[0,1]$ and the classes sum to 1. |
| Sigmoid vs softmax | Sigmoid is naturally binary or independently activated; softmax creates one shared multi-class distribution. |
| Cross-entropy vs accuracy | Cross-entropy uses confidence; accuracy only asks whether the top class was correct. |
| Loss vs gradient | Loss says how wrong; gradient says which direction to change the parameters. |
| One-hot label vs class index | One-hot stores the target as a vector; a class index stores one integer naming the class. |

---

## 19. What We Discovered

1. A multi-class model needs a **shared probability distribution**, not three unrelated scores.
2. Exponentials make arbitrary logits positive and preserve their ordering.
3. Normalization turns those positive values into probabilities: this is softmax.
4. Logarithms make confident mistakes increasingly expensive.
5. Cross-entropy is the natural negative log-likelihood for a categorical target.
6. The combined gradient is simply prediction minus truth.

---

## 20. Mathematics We Built

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$$
p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}},\qquad m=\max_j z_j
$$

$$
L=-\sum_i y_i\log p_i
$$

$$
L=\log\sum_j e^{z_j}-z_y
$$

$$
\frac{\partial L}{\partial z_i}=p_i-y_i
$$

---

## 21. What Each Symbol Means

| Symbol | Meaning | In code |
|---|---|---|
| $z_i$ | score/logit for class $i$ | `logits[i]` |
| $p_i$ | predicted probability | `probs[i]` |
| $y_i$ | one-hot target component | `target[i]` |
| $K$ | number of classes | `num_classes` |
| $m$ | maximum logit used for stability | `logits.max()` |
| $L$ | loss for one example | `loss` |

---

## 22. One-Minute Explanation

The model first makes one raw score for every class. Softmax converts those scores into probabilities by exponentiating and then dividing by the total. Cross-entropy looks only at the probability assigned to the true class and applies a negative logarithm. A correct confident prediction gets a tiny loss; a confident wrong prediction gets a large loss. Most importantly, the gradient is prediction minus truth, so learning has an immediate direction.

---

## 23. Exercises

### Level 1 — Observe
For logits `[1, 1, 1]`, before running code, describe the softmax probabilities.

### Level 2 — Calculate
Compute the cross-entropy when the true class probability is $0.25$.

### Level 3 — Derive
Starting from

$$
L=-\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right),
$$

derive the log-sum-exp form.

### Level 4 — Investigate
In the notebook, multiply every logit by $0.5$, then by $2$. Predict how the probability distribution changes before running the experiment.

### Level 5 — Design
Design a temperature parameter $T>0$ that makes softmax more or less confident. Decide which formula should approach a uniform distribution as $T\to\infty$.

---

## 24. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Calling logits probabilities | Logits are unconstrained real numbers. |
| Forgetting the denominator | The outputs no longer form a distribution. |
| Computing `exp(logits)` without stabilization | Large values can overflow. |
| Thinking cross-entropy cares equally about every probability | For one-hot targets it directly selects the true-class probability. |
| Confusing “most probable” with “well calibrated” | A model can choose the correct class while assigning badly distorted probabilities. |

---

## 25. Socratic Questions

Why does softmax need a denominator at all?

Why is adding the same constant to every logit harmless?

Why should a loss punish a confident wrong prediction more than an uncertain wrong prediction?

Why does the derivative of softmax cross-entropy collapse to `prediction - truth`?

Why might accuracy say two models are equally good when cross-entropy says they are very different?

---

## 🔭 Bridge to Chapter 16

We can now classify a house among several classes with a clean probability distribution. But notice what the model still assumes: its decision is built from a fixed mathematical rule applied to every point. What if we refuse to learn a global boundary at all?

What if the simplest possible classifier is: **find the houses most like this one, and let their labels vote?**

That question leads to nearest neighbours — and to a surprising problem: in very high dimensions, the idea of “near” starts to break.