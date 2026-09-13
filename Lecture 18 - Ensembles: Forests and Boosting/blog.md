# Lecture 18 — Ensembles: Forests and Boosting

> **The Big Question:** If one model can be fooled, can many imperfect models learn to make a better decision together?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2018%20-%20Ensembles%3A%20Forests%20and%20Boosting/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We just learned **Ensembles: Forests and Boosting**.

But using it creates a new question.

> **What problem does this idea still leave us unable to solve?**

That question leads naturally to **Support Vector Machines and Kernels**.

**Next → Support Vector Machines and Kernels.**

## 1. The Problem: One Tree Changes Its Mind

Suppose we classify houses as **premium** or **ordinary** using two features:

| House | Rooms | Area | Premium? |
|---|---:|---:|---|
| A | 2 | 700 | No |
| B | 2 | 900 | No |
| C | 3 | 1000 | Yes |
| D | 3 | 1200 | Yes |
| E | 4 | 1300 | Yes |
| F | 4 | 1500 | Yes |

A decision tree may choose a split near `area = 950` and get everything right.

Now one new training example arrives:

| G | 3 | 940 | No |

The best first split might move to 940. That tiny change can rearrange many downstream questions.

A tree is a **high-variance learner**: small changes in the training data can produce a noticeably different tree.

> 🧠 **Think** — A single tree asks a chain of yes/no questions. If the first question changes, every question below it may change too. How could we make the final answer less dependent on one chain?

---

## 2. What Would a Solution Need?

A useful ensemble should:

1. make different mistakes across models;
2. combine those models without simply copying one of them;
3. reduce instability when the training sample changes;
4. improve accuracy without requiring one giant, perfectly tuned model.

The key word is **diversity**. If ten models always make the same mistake, ten votes are still one mistake.

---

## 3. First Attempt: Ask the Same Tree Ten Times

Train exactly the same tree ten times on exactly the same data, with exactly the same settings.

The predictions are identical:

```text
Tree 1  → Yes
Tree 2  → Yes
...
Tree 10 → Yes
```

Majority vote adds no new information.

> ⚠️ **A Tempting Wrong Idea**
>
> *“More models must mean a better model.”*
>
> Not necessarily. If the models are clones, averaging them cannot remove their shared error. An ensemble becomes useful when its members contain useful **different** errors.

So we need a controlled way to make models different.

---

## 4. The Discovery: Bootstrap Samples

Take the training set and sample from it **with replacement**.

Imagine the original houses are:

```text
A B C D E F
```

A bootstrap sample might be:

```text
A C C D F F
```

Another might be:

```text
B B C D E F
```

Each tree sees a slightly different dataset. Because the data differ, the learned trees differ.

Why “with replacement”? After drawing `C`, we put `C` back before the next draw. A house can appear twice, while another house may be absent.

This simple trick is called **bagging**, short for bootstrap aggregating.

### Three levels

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Ask several students to solve the problem using slightly different homework sheets, then combine their answers. |
| ✏️ **Numbers** | Three trees vote `Yes, Yes, No` → majority is `Yes`. |
| 🎓 **Abstraction** | Draw bootstrap datasets $D^{(1)},\dots,D^{(B)}$, train models $f_1,\dots,f_B$, then aggregate their predictions. |

For regression, a natural aggregation is an average:

$$
\boxed{\hat y(x)=\frac{1}{B}\sum_{b=1}^{B}f_b(x)}
$$

For classification, we can use majority vote.

---

## 5. Why Averaging Can Help

Suppose each tree's prediction contains an error:

$$
\hat y_b = y + \varepsilon_b.
$$

The average prediction is

$$
\bar y = y + \frac{1}{B}\sum_{b=1}^{B}\varepsilon_b.
$$

If the errors were independent, their positive and negative parts would partly cancel.

For equal-variance independent errors,

$$
\mathrm{Var}\left(\frac{1}{B}\sum_b\varepsilon_b\right)
=\frac{\sigma^2}{B}.
$$

This is the heart of **variance reduction**.

Real trees are not independent, so the reduction is not exactly $\sigma^2/B$. But the principle survives: less-correlated errors are easier to average away.

> 💡 The ensemble does **not** make every tree wiser. It makes the final answer less sensitive to the quirks of one tree.

---

## 6. From Bagging to Random Forests

Bagging trees already creates diversity by changing the rows of the training set. Random forests add another source of diversity: at each split, consider only a **random subset of features**.

Suppose our data has 10 features.

Instead of allowing every tree to consider all 10 at every split, a split might be allowed to inspect only 3 randomly selected features.

Now two trees can see the same training example and still choose different questions because they are offered different candidate features.

```text
                 Training data
                      |
              bootstrap samples
              /       |       \
           Tree 1   Tree 2   Tree 3
             |        |        |
        random      random   random
        features    features features
              \       |       /
                majority vote
                      |
                  prediction
```

This is the **random forest** idea: many randomized decision trees whose predictions are aggregated.

---

## 7. A Tiny Forest by Hand

Suppose five trees classify a new house:

| Tree | Prediction |
|---|---|
| 1 | Premium |
| 2 | Ordinary |
| 3 | Premium |
| 4 | Premium |
| 5 | Ordinary |

Premium receives 3 votes; ordinary receives 2.

So:

$$
\boxed{\text{Forest prediction} = \text{Premium}}
$$

For regression, imagine predictions of ₹10, 12, 11, 13, 14 lakh:

$$
\hat y = \frac{10+12+11+13+14}{5}=12
$$

One tree said 10 and one said 14. The forest says 12.

---

## 8. A Different Problem: What If Every Tree Is Weak?

Bagging attacks **instability**. But what if each individual model is only slightly better than guessing?

Suppose three small models make these errors:

```text
Model 1: gets A B C right, D wrong
Model 2: gets A B D right, C wrong
Model 3: gets A C D right, B wrong
```

The mistakes are different. That is promising.

Can we train the next model to pay extra attention to the examples previous models got wrong?

That question leads to a completely different ensemble strategy.

---

## 9. The Discovery: Boosting

Start with a simple model.

Find where it performs badly.

Give those difficult examples more influence.

Train the next model.

Repeat.

The models are no longer independent voters. They form a **sequence**, where later models focus on correcting earlier mistakes.

At a high level:

```text
weak model 1
     ↓
look at residual mistakes
     ↓
weak model 2 focuses on them
     ↓
new residual mistakes
     ↓
weak model 3
     ↓
combine all weak learners
```

This is **boosting**.

---

## 10. Residuals Reveal the Idea

For regression, the most concrete version is to fit the residuals.

Start with a first model $f_1(x)$.

Its residual is:

$$
 r_i = y_i - f_1(x_i).
$$

Now train a second model $g_2(x)$ to predict those residuals.

Then update:

$$
 f_2(x)=f_1(x)+\eta g_2(x)
$$

where $\eta$ is a small **learning rate**.

Then compute new residuals and repeat:

$$
\boxed{f_M(x)=f_1(x)+\eta g_2(x)+\cdots+\eta g_M(x)}
$$

The model is built **additively**: each new learner adds a correction.

---

## 11. One Tiny Boosting Calculation

Suppose the true house prices are:

$$
[10,\;12,\;14]
$$

The first model predicts:

$$
[11,\;11,\;11]
$$

Residuals:

$$
[-1,\;1,\;3].
$$

The second model predicts those residuals approximately as:

$$
[-1,\;1,\;2].
$$

Take learning rate $\eta=0.5$.

Updated prediction:

$$
[11,11,11]+0.5[-1,1,2]
=[10.5,11.5,12].
$$

The errors become smaller for the first two examples, but the third still needs work.

That is the rhythm of boosting: **fit what is still wrong**.

---

## 12. Random Forests and Boosting Are Not the Same Trick

| Idea | Random forest | Boosting |
|---|---|---|
| Main goal | reduce variance | reduce bias / build a strong predictor from weak learners |
| Relationship between models | mostly parallel | sequential |
| Diversity comes from | bootstrap rows + random feature subsets | later models focus on earlier mistakes |
| Combination | vote / average | weighted additive combination |
| Typical base learners | decision trees | often shallow decision trees |

Neither is universally “better.” The right choice depends on data, noise, compute, tuning and objective.

---

## 13. What Does “Weak Learner” Actually Mean?

A weak learner is not necessarily a terrible model.

In the classic boosting intuition, a weak learner is one that performs only a little better than a suitable baseline on the training distribution.

The surprising result is that many such learners, combined carefully, can form a strong predictor.

That is one of the most beautiful recurring ideas in machine learning:

> **Structure can emerge from many small corrections.**

---

## 📜 History Lens — Breiman and Freund–Schapire

Imagine you are **Leo Breiman in the 1990s**, asking why unstable decision trees can become much more reliable when averaged. In 1996, Breiman formalized **bagging predictors**, showing how bootstrap aggregation can stabilize predictors with high variance.

Now imagine **Yoav Freund and Robert Schapire** asking a different question in the same decade: can a weak learning process be turned into a strong one? Their work on **AdaBoost** made boosting a practical and influential framework.

Later, boosting evolved into gradient boosting, where the next learner approximates the negative gradient of the loss rather than merely “counting mistakes.” That creates a bridge back to the calculus of Chapters 6–8.

The important historical split is therefore conceptual:

**bagging asks:** “How do I make unstable learners less unstable?”

**boosting asks:** “How do I keep adding corrections that improve the current model?”

---

## The Geometry

A decision tree partitions the feature space into regions.

A forest overlays many such partitions. A boosting model adds many small partition-based corrections.

The resulting decision boundary can be much more intricate than a single tree, while still being built from simple pieces.

Think of a forest as **many maps voting on the same territory** and boosting as **drawing one map, then repeatedly correcting its mistakes**.

---

## 🔬 The Experiment

Change exactly one variable: the number of trees.

Predict first:

> As we increase the number of randomized trees, what should happen to the forest's validation accuracy? Should it always increase? What about training time?

Then run the notebook. The important observation is not just the final score; it is whether additional trees continue to buy meaningful improvement.

---

## How It Breaks

| Failure | What it looks like | Why it happens |
|---|---|---|
| Correlated trees | many trees make the same error | the ensemble lacks diversity |
| Too-small trees | high bias | each learner cannot capture enough structure |
| Very deep trees | low training error, possibly poor generalization | individual trees can memorize |
| Boosting too aggressive | validation performance worsens | later learners chase noise |
| Data leakage | suspiciously excellent validation score | information from the future leaked into features |

---

## Shapes

For $n$ examples and $d$ features:

```text
X             : (n, d)
y             : (n,)
forest output : (n,)       classification labels
forest regress: (n,)       regression predictions
```

For a bootstrap sample of $n$ rows:

```text
X_bootstrap : (n, d)
```

A tree consumes one row at a time, but the ensemble combines the predictions from $B$ trees:

```text
predictions : (B, n)
aggregate   : (n,)
```

Shape alone tells us why averaging across the tree axis produces one prediction per example.

---

## 🎯 Machine Learning Connection

Ensembles matter because real datasets rarely have one perfectly correct view of the world.

Random forests are especially useful when you want a strong baseline with modest feature engineering. Boosting is often powerful when careful tuning and a structured feature table are available.

Modern gradient-boosting systems such as XGBoost and LightGBM are engineering descendants of the same additive idea.

---

## Distinctions That Matter

| Confusable pair | The real distinction |
|---|---|
| Bagging vs boosting | parallel diversity vs sequential correction |
| Random forest vs one tree | many diversified trees vs one fitted tree |
| averaging vs learning | combining fixed predictions vs changing the model |
| weak learner vs bad learner | deliberately simple learner vs useless learner |
| variance reduction vs bias reduction | stabilizing sensitivity vs correcting systematic underfitting |

---

## What We Discovered

1. A single tree can be unstable.
2. Different models are useful only when their errors are not identical.
3. Bagging creates diversity through bootstrap samples.
4. Random forests add random feature selection to diversify trees further.
5. Boosting builds an additive model that focuses on what remains wrong.
6. Ensembles succeed because simple learners can cooperate.

---

## Mathematics We Built

Bootstrap aggregation for regression:

$$
\hat y(x)=\frac{1}{B}\sum_{b=1}^{B}f_b(x)
$$

Residual:

$$
 r_i = y_i - f(x_i)
$$

One boosting update:

$$
 f_{m+1}(x)=f_m(x)+\eta g_{m+1}(x)
$$

Variance of an average of independent equal-variance errors:

$$
\mathrm{Var}(\bar\varepsilon)=\frac{\sigma^2}{B}
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $B$ | “bee” | number of ensemble members | `n_estimators` |
| $f_b$ | “f sub b” | prediction function of tree/model $b$ | `models[b]` |
| $\hat y$ | “y hat” | ensemble prediction | `prediction` |
| $\varepsilon$ | “epsilon” | prediction error | `error` |
| $r_i$ | “r sub i” | residual for example $i$ | `residual[i]` |
| $\eta$ | “eta” | boosting learning rate | `learning_rate` |

---

## One-Minute Explanation

A single decision tree can change a lot when the data changes a little. So build many different trees and let them vote or average. That is the idea behind random forests.

Boosting takes a different path: build a model, look at what it still gets wrong, train another small model to correct those errors, and add the correction. Repeat.

The big lesson is not “many models are always better.” It is that **different models can cooperate when their errors contain complementary information**.

---

## Exercises

### Level 1 — Observe
A forest has predictions `Yes, No, Yes, Yes, No`. What is the majority vote?

### Level 2 — Calculate
Average the regression predictions `[8, 10, 11, 9]`.

### Level 3 — Derive
Starting from $\hat y_b=y+\varepsilon_b$, derive the variance of the average when errors are independent and have variance $\sigma^2$.

### Level 4 — Investigate
In the notebook, change only `n_estimators` from 5 to 50 to 200. Plot validation score and training time. Predict the shape before running it.

### Level 5 — Design
Design an ensemble for a dataset with 1 million rows and 1,000 features where training one deep tree is expensive. Explain where you would introduce diversity and why.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| “More trees always means better predictions.” | Returns eventually diminish and computation increases. |
| “Random forest and boosting are the same.” | Their model-building process is fundamentally different. |
| “Boosting just votes.” | Boosting usually forms an additive weighted model. |
| “A weak learner is useless.” | Its usefulness can come from being complementary to other learners. |
| “Variance falls exactly by $1/B$ in every forest.” | That exact result assumes independent errors; real trees are correlated. |

---

## Socratic Questions

Why does averaging help only when model errors are at least partly different?

If every tree in a forest sees exactly the same features and data, what source of diversity remains?

Why might making a boosting learner deeper improve training error but hurt validation performance?

Why is a small learning rate often paired with more boosting rounds?

When would you prefer a forest over boosting, even if boosting achieves a slightly better validation score?

---

## 🔭 Bridge to Chapter 19

Ensembles can build complicated decision boundaries from trees. But suppose we want a classifier that is not defined by a tree partition at all. Could we choose a separating boundary that is **maximally safe** from the training points?

That question leads to a new geometric idea: the **margin**.

And once we understand the margin, we will discover why support vector machines can use kernels to create nonlinear boundaries without explicitly constructing all those new features.
