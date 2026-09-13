# Lecture 12 — Maximum Likelihood: Where Loss Functions Come From

> **The Big Question:** If a model is supposed to explain the data, how do we derive what it should try to maximize?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2012%20-%20Maximum%20Likelihood%3A%20Where%20Loss%20Functions%20Come%20From/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We just learned **Maximum Likelihood: Where Loss Functions Come From**.

But using it creates a new question.

> **What problem does this idea still leave us unable to solve?**

That question leads naturally to **Linear Regression, Properly**.

**Next → Linear Regression, Properly.**

## 1. The Problem: Two Coins, One Mystery

Imagine two coins:

- Coin A lands heads about half the time.
- Coin B is unknown. It might be fair, or heavily biased.

You toss Coin B **10 times** and see:

```text
H H H T H H T H H H
```

There are **8 heads** and **2 tails**.

Now ask the machine:

> **What value of the coin's head probability should we believe?**

Let that unknown probability be $p$.

If $p=0.5$, eight heads in ten tosses is possible.
If $p=0.8$, the observation looks much more natural.
If $p=0.99$, eight heads and two tails starts looking surprisingly unlikely again.

So neither “pick the middle” nor “pick the biggest $p$” works. We need a rule that asks:

> **Which $p$ makes the data we observed most plausible?**

That is the question that forces maximum likelihood.

---

## 2. What Would a Solution Need?

Our rule should:

1. Use the **observed data**, not imaginary future data.
2. Give us a way to **compare candidate parameter values**.
3. Reward parameters that make the observed data plausible.
4. Return the parameter value that is **best according to that comparison**.

Notice what we have *not* said. We have not chosen a loss function yet.

That comes later.

---

## 3. First Attempt: Just Choose the Most Frequent Value

We saw 8 heads and 2 tails. So perhaps simply say:

$$
p = 0.8
$$

For this tiny example, that answer is actually correct.

But why?

And what happens when the observations have different probabilities, as they will in real models?

The frequency is not a general principle. It is the answer for this particular likelihood model.

> ⚠️ **A Tempting Wrong Idea**
>
> *“Maximum likelihood just means matching the sample frequency.”*
>
> Frequency matching is a consequence of a **Bernoulli likelihood**. The deeper principle is broader: choose parameters that maximize the probability assigned to the observed data.

We need to write that principle mathematically.

---

## 4. The Discovery: Likelihood

For one coin toss,

$$
P(X=1\mid p)=p,
\qquad
P(X=0\mid p)=1-p.
$$

Suppose our data are $D=\{x_1,\ldots,x_n\}$ and each observation is modeled with parameter $\theta$.

If observations are conditionally independent given $\theta$, their joint probability is

$$
P(D\mid\theta)=\prod_{i=1}^{n}P(x_i\mid\theta).
$$

Now hold the data fixed and let the parameter move.

The same mathematical expression is called the **likelihood**:

$$
\boxed{L(\theta\mid D)=P(D\mid\theta)}
$$

The data are no longer the thing changing. **The parameter is.**

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Try different parameter dials. Ask: “With these dial settings, how believable is the dataset I actually saw?” |
| ✏️ **Numbers** | For 8 heads and 2 tails: $L(p)=p^8(1-p)^2$. At $p=0.5$, $L=0.01953125$. At $p=0.8$, $L=0.8^8 0.2^2\approx0.02684$. |
| 🎓 **Abstraction** | $L(\theta\mid D)=\prod_i P(x_i\mid\theta)$. **Maximum likelihood estimation** chooses $\hat\theta=\arg\max_\theta L(\theta\mid D)$. |

The notation $\arg\max$ means: **the parameter value at which the expression is largest**.

---

## 5. Why Products Become a Problem

The likelihood is a product.

For ten observations, that is manageable. For one million observations, the product can become an extremely tiny number.

For independent data:

$$
L(\theta)=\prod_{i=1}^{n}P(x_i\mid\theta).
$$

A product is also awkward to differentiate.

So we ask:

> Can we transform the likelihood without changing which parameter value is best?

Yes.

Take the logarithm.

Because $
\log
$ is strictly increasing,

$$
\arg\max_\theta L(\theta)
=
\arg\max_\theta \log L(\theta).
$$

And the logarithm turns products into sums:

$$
\log L(\theta)
=
\sum_{i=1}^{n}\log P(x_i\mid\theta).
$$

This is the **log-likelihood**.

> 💡 **Why this is more than a numerical trick**
>
> The logarithm preserves the ordering of candidates while changing multiplication into addition. That gives us both numerical stability and much simpler derivatives.

---

## 6. The Coin Derivation

For a Bernoulli observation $x_i\in\{0,1\}$,

$$
P(x_i\mid p)=p^{x_i}(1-p)^{1-x_i}.
$$

Therefore

$$
L(p)=\prod_{i=1}^{n}p^{x_i}(1-p)^{1-x_i}.
$$

Take logs:

$$
\ell(p)=\log L(p)
=\sum_{i=1}^{n}\left[x_i\log p+(1-x_i)\log(1-p)\right].
$$

Differentiate:

$$
\frac{d\ell}{dp}
=
\sum_i\left[\frac{x_i}{p}-\frac{1-x_i}{1-p}\right].
$$

For our data, $\sum_i x_i=8$ and $n=10$:

$$
\frac{d\ell}{dp}
=
\frac{8}{p}-\frac{2}{1-p}.
$$

At the optimum the derivative is zero:

$$
\frac{8}{p}=\frac{2}{1-p}.
$$

Cross-multiply:

$$
8(1-p)=2p.
$$

So

$$
8=10p
\qquad\Rightarrow\qquad
\boxed{\hat p=0.8}.
$$

The observed frequency did not appear by magic. **Maximum likelihood derived it.**

---

## 7. From Maximizing Likelihood to Minimizing Loss

Machine-learning code often minimizes a loss rather than maximizes a likelihood.

These are the same optimization written in opposite directions.

Define the negative log-likelihood:

$$
\boxed{\mathcal L(\theta)=-\log L(\theta)}.
$$

Then

$$
\arg\max_\theta L(\theta)
=
\arg\min_\theta[-\log L(\theta)].
$$

For independent data,

$$
\mathcal L(\theta)
=
-\sum_i\log P(x_i\mid\theta).
$$

This is the key bridge:

```text
probability model
      ↓
 likelihood of observed data
      ↓  take log
 log-likelihood
      ↓  multiply by −1
 loss / negative log-likelihood
      ↓
 gradient descent
```

> 🎯 **Machine Learning Connection**
>
> A loss function does not have to be invented by taste. Once you specify a probabilistic model for the data, **negative log-likelihood gives a principled objective**.

---

## 8. The Gaussian Surprise: Why Squared Error Appears

Now use a different data story.

Suppose a house price is modeled as

$$
\hat y_i = w x_i+b
$$

and the errors are assumed to be Gaussian with mean zero and standard deviation $\sigma$:

$$
y_i\mid x_i \sim \mathcal N(\hat y_i,\sigma^2).
$$

The density is

$$
P(y_i\mid x_i,w,b)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(y_i-\hat y_i)^2}{2\sigma^2}\right).
$$

Take the negative log and discard terms that do not depend on $w,b$:

$$
-\log P(y_i\mid x_i,w,b)
=
\text{constant}
+
\frac{(y_i-\hat y_i)^2}{2\sigma^2}.
$$

Summing over all observations gives

$$
\boxed{
\mathcal L(w,b)
=
\text{constant}
+
\frac{1}{2\sigma^2}
\sum_i(y_i-wx_i-b)^2
}
$$

Therefore minimizing negative log-likelihood is equivalent to minimizing **squared error**.

This is the deeper lesson:

> **Squared error is not a law of nature. It is the negative log-likelihood that appears when we assume Gaussian noise.**

Change the probability model, and the loss can change too.

---

## 9. The Geometry of the Objective

For a linear model, parameters such as $w$ and $b$ define a point in parameter space.

The loss assigns every point one height.

```text
loss
 ^
 |                 •
 |            •         •
 |       •                 •
 |  •                           •
 +---------------------------------> parameter
                 minimum
```

Maximum likelihood says:

> Find the parameter setting where the observed data receives the highest probability.

Negative log-likelihood says the same thing as a valley:

> Find the parameter setting where the loss is smallest.

---

## 🔬 The Experiment

Use the notebook to vary one thing at a time:

1. Keep the 8-head/2-tail dataset fixed and sweep $p$ from 0.01 to 0.99.
2. Plot likelihood and negative log-likelihood.
3. Verify that both choose the same $p$.
4. Change the number of heads and observe the optimum move.

Predict first. Then run.

---

## How It Breaks

| Failure | What it looks like | Why it is wrong |
|---|---|---|
| Maximize raw probability forever | tiny underflowing products | numerical scale collapses |
| Treat likelihood as probability *of the parameter* | “$P(\theta\mid D)$” without a prior | that is a Bayesian quantity, not ordinary likelihood |
| Forget independence assumptions | multiply incompatible terms | joint probability may not factor this way |
| Assume squared error is universal | use it for every target type | the implied noise model matters |
| Ignore constants carelessly | claim two objectives are always numerically equal | they are often equivalent only up to parameter-independent constants/scales |

---

## Shapes

For $n$ independent scalar observations:

```text
x       : (n,)
y       : (n,)
params  : depends on model
loglik  : scalar
loss    : scalar
```

The important shape fact is simple:

> **A dataset can contain many observations, but the objective is one scalar.**

That scalar is what gradient descent knows how to optimize.

---

## Distinctions That Matter

| Confusable pair | Difference |
|---|---|
| Probability vs likelihood | Probability varies with possible data while parameters are fixed; likelihood treats observed data as fixed and compares parameters. |
| Likelihood vs posterior | Likelihood is $P(D\mid\theta)$; posterior is proportional to $P(D\mid\theta)P(\theta)$. |
| Log-likelihood vs likelihood | Same optimum because log is strictly increasing, but different numerical scale. |
| Loss vs negative log-likelihood | Negative log-likelihood is a loss; not every hand-designed loss is automatically an NLL. |

---

## What We Discovered

1. A model can be judged by how plausible it makes the data we actually observed.
2. That comparison is the likelihood.
3. Maximizing likelihood gives a general parameter-estimation rule.
4. Taking logs turns products into sums without changing the best parameter.
5. Minimizing negative log-likelihood turns statistical estimation into the optimization language of machine learning.
6. Squared error appears naturally from a Gaussian observation model.

---

## Mathematics We Built

$$
L(\theta\mid D)=P(D\mid\theta)
$$

$$
\hat\theta_{MLE}=\arg\max_\theta L(\theta\mid D)
$$

$$
\ell(\theta)=\log L(\theta)
$$

$$
\arg\max \ell(\theta)=\arg\max L(\theta)
$$

$$
\mathcal L(\theta)=-\ell(\theta)
$$

$$
\hat\theta_{MLE}=\arg\min_\theta\mathcal L(\theta)
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $D$ | “data” | observed examples | `data` |
| $\theta$ | “theta” | model parameters | `theta` |
| $L$ | “likelihood” | plausibility of observed data under $\theta$ | `likelihood` |
| $\ell$ | “ell” | log-likelihood | `log_likelihood` |
| $\mathcal L$ | “script L” | negative log-likelihood loss | `loss` |
| $\hat\theta$ | “theta-hat” | estimated parameter | `theta_hat` |

---

## One-Minute Explanation

Suppose you have a model with a few adjustable knobs. You already saw some data. Try a candidate setting of the knobs and ask: **how well would these settings explain the exact data I observed?** That score is the likelihood. Pick the setting with the largest score. Taking a log makes products easier to work with, and putting a minus sign turns “make it large” into the usual machine-learning instruction “make the loss small.” Different assumptions about how data are generated lead to different losses.

---

## Exercises

### Level 1 — Observe
Look at the likelihood curve for 8 heads and 2 tails. Where is its peak, and why should the peak not be at $p=0.5$?

### Level 2 — Calculate
Compute $L(p)$ for $p=0.2$, $0.5$, and $0.8$ for the sequence with 8 heads and 2 tails.

### Level 3 — Derive
Starting from $\ell(p)=8\log p+2\log(1-p)$, derive $\hat p=0.8$.

### Level 4 — Investigate
Change the data to 3 heads and 7 tails. Predict the MLE before running the notebook.

### Level 5 — Design
Invent a small observation model for another problem where you can derive the corresponding negative log-likelihood. Explain what assumption about noise or outcomes you made.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Saying “likelihood is the probability that the parameter is true” | That changes $P(D\mid\theta)$ into a different quantity. |
| Dropping constants before identifying what depends on parameters | Constants can be dropped only when optimizing over parameters and only if they truly do not depend on them. |
| Thinking the log changes the optimum | A strictly increasing transform preserves ordering. |
| Treating every loss as an MLE | Some losses are design choices or arise from different assumptions. |

---

## Socratic Questions

- Why does the logarithm preserve the maximizing parameter?
- What would change if the observations were not independent?
- Why does a Gaussian assumption lead to squared error rather than absolute error?
- What extra ingredient would turn likelihood into a posterior?

---

## 🔭 Bridge to Chapter 13

We can now **derive** an objective instead of merely choosing one.

Next question:

> **What happens when we fit a real linear model to many observations and let maximum likelihood choose its parameters?**

That is where linear regression stops being a memorized formula and becomes a complete statistical model.