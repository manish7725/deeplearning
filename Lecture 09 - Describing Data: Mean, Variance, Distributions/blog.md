# Lecture 09 — Describing Data: Mean, Variance, Distributions

> **The Big Question:** Real measurements scatter. How do we describe that scatter with a handful of numbers?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2009%20-%20Describing%20Data%3A%20Mean%2C%20Variance%2C%20Distributions/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We just learned **Describing Data: Mean, Variance, Distributions**.

But using it creates a new question.

> **What problem does this idea still leave us unable to solve?**

That question leads naturally to **Maximum Likelihood: Where Loss Functions Come From**.

**Next → Maximum Likelihood: Where Loss Functions Come From.**

## 1. The Problem: Ten Identical Houses, Ten Different Prices

The office pulls the records for every 3-room, 900 sq ft flat sold in the same street last year. Ten sales, ten *identical* properties on paper. Here is what they went for, in ₹ lakh:

$$
9,\quad 10,\quad 11,\quad 11,\quad 12,\quad 12,\quad 12,\quad 13,\quad 14,\quad 16
$$

Identical houses. Prices from 9 to 16 — a spread of seven lakh, nearly double from bottom to top.

Nothing in our model can express this. Chapter 2's model says $\hat y = \mathbf{w}\cdot\mathbf{x} + b$: feed it the same $\mathbf{x}$ and it returns the same $\hat y$, every time. Chapter 2 §1 showed that when two houses with the same input have different outputs, **no values of $w$ and $b$ can fit both** — we used that to argue for more features. But here the inputs really are identical in every way the office records. There is no missing column.

The variation is not a modelling failure. It is what actually happens: one seller was in a hurry, one buyer fell in love with the balcony, one sale closed the week the new metro line was announced.

> 🧠 **Think** — Chapter 8 could drive the training loss to *exactly zero*, because our four houses sat exactly on a line. On this data, zero loss is not just unlikely, it is **wrong**. A model that passes through all ten of these points has memorized ten accidents. Chapter 1 §15 called this overfitting and promised to come back to it.

So the manager asks two questions, and we cannot currently answer either:

1. **What should we quote** for the next identical flat?
2. **How confident should we be?** A quote of ₹12 lakh means something very different if prices range 11.9–12.1 than if they range 9–16.

---

## 2. What Would a Summary Need?

1. **Collapse many numbers into few.** Ten numbers today, ten million later.
2. **Say what is typical** — a single best guess.
3. **Say how much things vary** — because question 2 above is about spread, not centre.
4. **Use the same units** as the data, so the answer is interpretable in ₹ lakh.
5. **Survive scaling up.** Whatever we compute for ten sales must work identically for ten million.

Requirements 2 and 3 are genuinely different questions, and §5 shows what goes wrong when you answer only the first.

---

## 3. First Attempt: What Is the Typical Price?

Pick one sale and quote it? Which one — the ₹9 lakh or the ₹16 lakh? Any single sale is an accident.

The instinct almost everyone has is right: **share the total equally**. If all ten houses had sold for the same price, what would that price be?

$$
\text{total} = 9+10+11+11+12+12+12+13+14+16 = 120
$$
$$
\text{shared equally} = \frac{120}{10} = 12
$$

That is the **mean**, and notice that we derived it rather than defining it. It answers: *if the same total money had been spread evenly over the same number of houses, what would each have cost?*

Now generalize. With $n$ values $x_1, \ldots, x_n$:

$$
\bar{x} = \frac{x_1 + x_2 + \cdots + x_n}{n}
$$

and since we will be writing sums constantly from here on, compress it with the sigma notation Chapter 1 §8 introduced:

$$
\boxed{\;\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i\;}
$$

Read the sigma aloud: *"start at the first house, add its price, keep going, stop at house $n$"* — then divide by how many there were. The bar over $x$ is the standard mark for "the mean of".

| Level | The same idea |
|---|---|
| 💡 **Intuition** | The balance point. Put one unit of weight at each price along a ruler; the mean is where the ruler balances. |
| ✏️ **Numbers** | $120 \div 10 = 12$ ₹ lakh. |
| 🎓 **Abstraction** | $\bar x = \frac1n\sum x_i$, the value that would give the same total if shared equally. |

Requirement 2 is satisfied. We quote **₹12 lakh**.

---

## 4. The Balance Point, and a Result We Have Seen Before

There is a property of the mean worth seeing directly, because it comes back immediately in §5 and it is not a coincidence.

Take each price and ask how far it sits from the mean. Call that its **deviation**, $x_i - \bar x$:

| price | 9 | 10 | 11 | 11 | 12 | 12 | 12 | 13 | 14 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deviation | $-3$ | $-2$ | $-1$ | $-1$ | $0$ | $0$ | $0$ | $+1$ | $+2$ | $+4$ |

Add the deviations up:

$$
(-3) + (-2) + (-1) + (-1) + 0 + 0 + 0 + 1 + 2 + 4 = \mathbf{0}
$$

Exactly zero. And not by luck — it is forced. Split the sum:

$$
\sum_{i=1}^{n}(x_i - \bar x) = \sum_i x_i - \sum_i \bar x = n\bar x - n\bar x = 0
$$

using the fact that $\sum x_i = n\bar x$, which is just the definition of the mean rearranged. **The deviations from the mean always sum to zero, for every dataset that has ever existed.** That is exactly what "balance point" means: the pull from below cancels the pull from above.

> ⚠️ Remember this the moment we try to measure spread. Chapter 1 §7 already hit this wall once, when averaging signed errors declared a useless model flawless. **The same trap is about to appear in a new costume.**

---

## 5. First Attempt at Spread: Average the Deviations

Requirement 3 asks how much prices vary. We have the deviations. Average them:

$$
\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar x) = \frac{0}{10} = 0
$$

**Zero spread.** According to this measure, our ten prices — ranging from 9 to 16 — do not vary at all.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Spread is the average distance from the middle, so average the deviations."*
>
> The intuition is right; the arithmetic is fatal. §4 proved the deviations *always* sum to zero, so this measure returns 0 for every dataset in existence. It is not merely inaccurate — it carries no information at all.
>
> This is precisely Chapter 1 §7, where a model quoting ₹6 lakh for every house scored a perfect average error of zero because $+3$ and $+1$ cancelled $-1$ and $-3$. Same disease, same cure needed: **get rid of the signs.**

Chapter 1 offered two ways to discard a sign, and they are available again.

**Take absolute values.** The **mean absolute deviation**:

$$
\frac{1}{n}\sum|x_i - \bar x| = \frac{3+2+1+1+0+0+0+1+2+4}{10} = \frac{14}{10} = 1.4
$$

Honest and interpretable: prices sit about ₹1.4 lakh from the middle on average.

**Square them.** And here, unlike Chapter 1, squaring is not merely a preference — it is about to earn its place in a way absolute values cannot match.

---

## 6. The Discovery: Variance

Square each deviation, then average:

| deviation | $-3$ | $-2$ | $-1$ | $-1$ | $0$ | $0$ | $0$ | $+1$ | $+2$ | $+4$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| squared | $9$ | $4$ | $1$ | $1$ | $0$ | $0$ | $0$ | $1$ | $4$ | $16$ |

$$
\sum(x_i-\bar x)^2 = 36
$$

This is the **variance**. Two versions exist and the difference matters:

$$
\sigma^2 = \frac{1}{n}\sum(x_i-\bar x)^2 = \frac{36}{10} = 3.6
\qquad
s^2 = \frac{1}{n-1}\sum(x_i-\bar x)^2 = \frac{36}{9} = 4.0
$$

### Why divide by $n-1$?

This looks like a fudge and is not. §4 proved the deviations must sum to zero — which means that once you know nine of them, **the tenth is fully determined**. There are ten numbers but only nine independent pieces of information about spread; one was spent computing $\bar x$ itself.

That count is called the **degrees of freedom**. Dividing by $n$ measures the spread of *these ten houses* and nothing more. Dividing by $n-1$ estimates the spread of the *whole street* from a sample of ten, correcting for the fact that the sample's own mean sits closer to the sample than the true mean would. Use $n-1$ when the data is a sample of something larger — which, in machine learning, it essentially always is.

### The standard deviation

Variance has a units problem. Our prices are in ₹ lakh, so squared deviations are in *squared lakh*, which means nothing to anybody. Requirement 4 says fix it — take the square root:

$$
\boxed{\;s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar x)^2}\;} = \sqrt{4.0} = 2.0
$$

**₹2 lakh.** Back in the original units, and now the manager's second question has an answer: *"about ₹12 lakh, give or take ₹2 lakh."*

> 💡 **Intuition** — The standard deviation is a typical distance from the mean. Not the largest, not the smallest — the usual one.

Compare it with Chapter 2 §8, which defined the length of a vector as $\|\mathbf{x}\| = \sqrt{\sum x_i^2}$. The standard deviation is that same formula applied to the deviation vector, then scaled. **Spread is the length of the deviation vector** — the geometry of Part I describing statistics, and the first sign that these two subjects are the same subject.

---

## 7. The Shape of the Scatter

The mean says where, the standard deviation says how far. Neither says what the scatter *looks like*. Group the prices into bins:

```text
 9  ■
10  ■
11  ■■
12  ■■■
13  ■
14  ■
16  ■
    └─────────────────────
    9  10  11  12  13  14  15  16
```

More sales near the middle, fewer at the edges, and a tail stretching further right than left. A picture of counts like this is a **histogram**, and its shape is the **distribution** of the data.

That hump-in-the-middle shape appears constantly — in heights, in measurement errors, in exam marks. The idealized version is the **normal distribution**, also called the **Gaussian**:

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Do not try to absorb that formula yet — Chapter 10 explains what $p(x)$ means and Chapter 12 uses it in anger. For now, read only the part in the exponent:

$$
-\frac{(x-\mu)^2}{2\sigma^2}
$$

**A squared deviation from the mean, divided by the variance, with a minus sign.** Values far from $\mu$ get a large squared deviation, hence a large negative exponent, hence a tiny probability. The bell shape is *built out of squared distance from the centre* — the exact quantity we invented in §6, and the exact quantity Chapter 1 chose for its loss.

> 🔭 That is not a coincidence, and it is the single most important thing to carry out of this chapter. Chapter 12 will show that **assuming Gaussian noise and then asking which parameters best explain the data produces squared error, necessarily.** Chapter 1's "choice" was a consequence of an assumption we did not know we were making.

### Why this shape keeps appearing

Briefly and honestly: when a quantity is the sum of many small independent effects, its distribution tends towards this bell shape regardless of what the individual effects look like. That statement is the **Central Limit Theorem**. A house price is exactly such a sum — the seller's haste, the buyer's mood, the weather, the metro announcement — which is why prices scatter this way.

Two honest caveats. The theorem needs the effects to be *independent* and none to dominate. When one effect dominates — a single billionaire in an income dataset — the result is not Gaussian at all, and §9 shows what that does to our summaries.

---

## 8. Two Variables: Covariance and Correlation

One more question the office asks: **do bigger houses cost more?** That is about two measurements moving together, and neither the mean nor the standard deviation can answer it.

Return to Chapter 2's four houses:

| | rooms | area (sq ft) |
|---|---:|---:|
| A | 2 | 800 |
| B | 2 | 1200 |
| C | 3 | 900 |
| D | 4 | 1600 |

Means: $\bar r = 2.75$ rooms, $\bar a = 1125$ sq ft.

The idea: when rooms is *above* its mean, is area usually above its mean too? Multiply the two deviations for each house — the product is positive when both are on the same side, negative when opposite — and average:

$$
\mathrm{cov}(r,a) = \frac{1}{n-1}\sum_{i}(r_i - \bar r)(a_i - \bar a) = 241.67
$$

Positive, so rooms and area rise together. But **241.67 what?** The units are room·sq ft, which is meaningless, and the number would change entirely if we measured area in square metres. Covariance has the same units problem variance had, and worse.

Fix it by dividing out both standard deviations:

$$
\boxed{\;r = \frac{\mathrm{cov}(x,y)}{s_x\,s_y}\;} = \frac{241.67}{\sqrt{0.9167}\cdot\sqrt{129166.7}} = \mathbf{0.70232}
$$

This is the **correlation coefficient**, and it is always between $-1$ and $+1$, with no units at all.

| $r$ | meaning |
|---:|---|
| $+1$ | perfectly in step: one rises exactly as the other does |
| $0$ | no *linear* relationship |
| $-1$ | perfectly opposed |

Our $0.70$ says rooms and area move together strongly but not perfectly — house B has few rooms and lots of area, which is what stops it being 1.

> 🎯 **You have computed this before.** Chapter 2 §9 defined cosine similarity as $\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$. Correlation is *exactly* that formula, applied to the two deviation vectors. Statistics has not invented anything new here — **correlation is the cosine of the angle between two centred variables**, and $r = 0.70$ means an angle of about $45°$.
>
> And Chapter 5 §10 already used this number: the correlation matrix whose eigenvalues gave PC1 $85.1\%$ of the variance had exactly $0.70232$ in its off-diagonal. Three chapters, one number, arrived at three different ways.

> ⚠️ **Correlation is not causation, and it is also not *any* relationship.** $r$ measures *straight-line* agreement only. Data lying perfectly on a parabola can have $r = 0$ while being perfectly predictable. A correlation of zero rules out a linear relationship and nothing else.

---

## 9. 🔬 The Experiment: One Number Changes Everything

> 🧠 **Predict before reading on.** Take our ten prices and change the largest, ₹16 lakh, into ₹60 lakh — one mansion accidentally included in the street's records. What happens to the mean, the median (the middle value when sorted), and the standard deviation? Which of the three survives?

The original ten, sorted: $9, 10, 11, 11, 12, 12, 12, 13, 14, 16$. The middle two are both 12, so the **median** is 12 — same as the mean, for this well-behaved data.

Now swap $16 \to 60$:

| | original | with the mansion |
|---|---:|---:|
| mean | $12.0$ | $\mathbf{16.4}$ |
| median | $12.0$ | $\mathbf{12.0}$ |
| standard deviation | $2.0$ | $\mathbf{15.4}$ |

The mean jumps by more than four lakh — to a value **higher than nine of the ten houses actually sold for**. Quote ₹16.4 lakh for the next flat and you will not sell it. The standard deviation explodes from 2 to 15, claiming the prices are wildly unpredictable when nine of them sit within ₹2.5 lakh of each other.

The **median does not move at all.**

The reason is visible in the formulas. The mean sums every value, so one huge number drags the total. Variance sums *squared* deviations, so a value 44 away contributes $44^2 = 1936$ — more than the other nine combined, by a factor of fifty. The median only asks *which value is in the middle*, so the size of the largest number is irrelevant.

> 💡 **Squaring's great strength is its great weakness.** Chapter 1 chose squared error precisely because it punishes large mistakes disproportionately. That is the same property that lets a single outlier dominate a variance — and, in Chapter 13, lets a single mislabelled house drag an entire fitted line.

---

## 10. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **Outliers** | mean and $s$ become meaningless | Both weight extremes heavily; squaring more so. §9 |
| **Averaging deviations** | spread reported as zero, always | They sum to zero by construction. §5 |
| **Quoting only the mean** | confident, frequently wrong | Two datasets can share a mean and have utterly different risk. §2 |
| **Dividing by $n$ on a sample** | spread underestimated | One degree of freedom went into $\bar x$. §6 |
| **Comparing covariances** | nonsense conclusions | Units-dependent; switch sq ft to sq m and it changes. §8 |
| **Reading $r$ as "related"** | missing obvious patterns | $r$ sees straight lines only. A perfect parabola gives $r=0$. §8 |
| **Two clusters** | mean lands where nothing exists | A bimodal dataset's mean can sit in the empty gap between the groups. |

---

## 11. 🎯 Machine Learning Connection

| This chapter | In machine learning |
|---|---|
| mean | the baseline every model must beat (Ch 20) |
| variance, $s$ | the noise floor — the loss no model can get below |
| dividing by $n-1$ | why sample statistics are corrected |
| Gaussian | the noise assumption that *derives* squared error (Ch 12) |
| correlation | multicollinearity, and feature selection (Ch 13, Ch 22) |
| outlier sensitivity | why squared error is fragile, and when to use MAE |
| standardizing $(x-\bar x)/s$ | feature scaling — Chapters 2 and 5 did this without naming it |

The most useful idea here is the **noise floor**. Our ten identical houses vary with a standard deviation of ₹2 lakh for reasons no feature captures. A model predicting the mean for all of them has a mean squared error of about $\sigma^2 = 3.6$. **No model can do better on this data**, because the remaining variation is not a function of anything we recorded.

So if your training loss falls below the noise floor, you are not learning — you are memorizing. Chapter 20 turns this into train/test splits, and Chapter 23 into regularization.

---

## 12. Distinctions That Matter

| | |
|---|---|
| **Mean** — balance point, sensitive to outliers | **Median** — middle value, robust |
| **Variance** $s^2$ — squared units | **Standard deviation** $s$ — original units |
| $\sigma^2$ (÷ $n$) — describes this data | $s^2$ (÷ $n-1$) — estimates the population |
| **Deviation** $x_i - \bar x$ — signed, sums to 0 | **Squared deviation** — positive, sums to $36$ |
| **Covariance** — has units, unbounded | **Correlation** — unitless, in $[-1, 1]$ |
| **Correlation** — straight-line agreement | **Dependence** — any relationship at all |
| **Data** $\bar x$, $s$ — computed from a sample | **Distribution** $\mu$, $\sigma$ — the ideal it came from |

---

## 13. What We Discovered

1. Identical houses sell for different prices, and no extra feature fixes it — real measurements scatter.
2. A model driven to exactly zero loss on scattered data has memorized accidents.
3. The mean is derived, not defined: the price each house would have cost if the total were shared equally.
4. Deviations from the mean always sum to exactly zero — that is what "balance point" means.
5. Therefore averaging deviations reports zero spread for *every* dataset — Chapter 1's cancellation trap, wearing new clothes.
6. Removing the sign by squaring gives variance; taking the root gives the standard deviation, back in original units.
7. Dividing by $n-1$ counts degrees of freedom: one was spent computing the mean.
8. The Gaussian's exponent is a squared deviation over the variance — the same quantity Chapter 1 chose as its loss, which Chapter 12 shows is no coincidence.
9. Correlation is Chapter 2's cosine similarity applied to centred variables — $0.70232$, the same number Chapter 5's PCA used.
10. One outlier moves the mean by 4.4 and the standard deviation by 13.4, and the median not at all.

## 14. Mathematics We Built

$$
\bar x = \frac1n\sum_{i=1}^n x_i
\qquad
\sum_{i=1}^n (x_i - \bar x) = 0
$$

$$
s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar x)^2
\qquad
s = \sqrt{s^2}
\qquad
\text{MAD} = \frac1n\sum|x_i - \bar x|
$$

$$
\mathrm{cov}(x,y) = \frac{1}{n-1}\sum (x_i-\bar x)(y_i-\bar y)
\qquad
r = \frac{\mathrm{cov}(x,y)}{s_x s_y} \in [-1, 1]
$$

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

## 15. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $\bar x$ | "x-bar" — the mean of the data | `x.mean()` |
| $\mu$ | "mu" — the mean of the true distribution | — |
| $s$ | sample standard deviation | `x.std(ddof=1)` |
| $\sigma$ | "sigma" — true standard deviation | — |
| $s^2$, $\sigma^2$ | variance | `x.var(ddof=1)` |
| $n-1$ | degrees of freedom | `ddof=1` |
| $x_i - \bar x$ | deviation of one value | `x - x.mean()` |
| $\mathrm{cov}(x,y)$ | how two variables move together | `np.cov(x, y)` |
| $r$ | correlation, unitless, in $[-1,1]$ | `np.corrcoef(x, y)` |

## 16. One-Minute Explanation

With no equations:

> Ten identical flats sold for ten different prices. What two numbers would you give the manager, and why is one of them useless on its own?

---

## 17. Exercises

**Level 1 — Observe.** Look at the histogram in §7. Is the scatter symmetric, or does one side stretch further? Which single sale is responsible? Then look at §9's table: explain, in one sentence each, why the mean moved, why the standard deviation moved *more*, and why the median did not move at all.

**Level 2 — Calculate (by hand, no code).** For the five prices $10, 12, 12, 14, 17$: compute the mean; write out all five deviations and confirm they sum to zero; compute the variance both ways ($\div n$ and $\div n-1$); take the square root of each. Then compute the mean absolute deviation and say which of MAD and $s$ is larger, and why that will *always* be true.

**Level 3 — Derive.** Prove $\sum_{i}(x_i - \bar x) = 0$ from the definition of $\bar x$. Then prove the computational shortcut
$$\frac1n\sum (x_i-\bar x)^2 = \frac1n\sum x_i^2 - \bar x^2$$
by expanding the square. Verify it on the ten prices ($\sum x_i^2 = 1476$). Finally: why might a computer prefer the left-hand form even though the right-hand one needs only one pass through the data? (Think about Chapter 6 §11 and subtracting nearly equal numbers.)

**Level 4 — Investigate** (notebook Steps 8–11). Draw 10 prices repeatedly from a Gaussian with $\mu = 12$, $\sigma = 2$, and compute the sample mean each time. The sample means scatter too — measure *their* standard deviation. Now repeat with samples of 40 and 160. The spread of the sample mean shrinks in a specific pattern: find it, and state how many sales you would need to pin the average price to within ₹0.1 lakh.

**Level 5 — Design.** The manager wants a single number for "how risky is this street?" that is not wrecked by one mansion. Design one. State precisely what it computes, verify it on §9's outlier data, and then say honestly what your measure *loses* compared with the standard deviation — because every robust statistic trades something away.

---

## 18. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "The mean is the most common value." | That is the *mode*. The mean can be a value that never occurs — like 11.5 for our data if one price changed. |
| "Standard deviation is the average distance from the mean." | Close but not equal: that is the MAD ($1.4$ here). The standard deviation is $2.0$, always larger. |
| "Divide by $n$, obviously." | On a sample that underestimates spread. One degree of freedom went into $\bar x$. §6 |
| "Variance is in the same units as the data." | It is in *squared* units. That is exactly why $s$ exists. |
| "$r = 0$ means no relationship." | It means no *linear* relationship. A perfect parabola scores $r=0$. §8 |
| "High correlation means one causes the other." | It means they move together. Ice-cream sales and drowning correlate; neither causes the other. |
| "Zero training loss is the goal." | On scattered data, zero loss means you fitted the noise. §1, §11 |

## 19. Socratic Questions

1. §4 proved the deviations sum to zero. Is there any dataset at all where they do not? What would that imply about $\bar x$?
2. We divide by $n-1$ because one degree of freedom went into $\bar x$. If a formula needed *two* quantities estimated from the data first, what would you divide by — and can you say why?
3. The Gaussian's exponent contains squared distance from the centre. Chapter 1 chose squared error for completely unrelated reasons. Is that a coincidence, or is one of them causing the other?
4. Correlation is the cosine of an angle between centred vectors (§8). What does $r = 0$ mean geometrically, and where have you seen that word before?
5. Our ten identical houses have a noise floor of $\sigma^2 = 3.6$. What would it mean for a model to reach a training loss of $0.5$ on this data — is that success?
6. The median ignored the mansion entirely. Why, then, is the mean used so much more often in machine learning? What does the median give up?

---

## 20. 🔭 Bridge to Chapter 10

We can now describe scatter. Ten prices collapse to two numbers — ₹12 lakh, give or take ₹2 lakh — plus a shape, and two variables collapse to a correlation. Requirements 1 through 5 are satisfied.

Then the manager asks a different kind of question entirely:

> *"A buyer has offered ₹14 lakh. What are the chances the next identical flat sells for more than that?"*

Look at what we have and notice that **none of it answers this**. The mean is 12 and the standard deviation is 2, so 14 is one standard deviation above the middle — but "one standard deviation above" is not a chance. We could count: exactly two of our ten sales exceeded 14, so perhaps 20%? But that is a statement about ten sales that already happened, and the manager is asking about one that has not.

Every tool in this chapter is **backward-looking**. It describes numbers we already have. The question is forward-looking: it asks about an event that has not occurred, and it wants a number between 0 and 1.

Worse, we cannot even state the question precisely. What *is* "the chance"? Not a count, because the sale has not happened. Not a prediction, because we will not be certain even after seeing the data. It is a new kind of quantity, and we have no mathematics for it.

> **What does it mean to assign a number to something that has not happened yet — and how do we compute with such numbers?**

That is where Chapter 10 begins. It builds the language of probability, which turns "how likely?" into arithmetic — and gives us the last vocabulary Chapter 12 needs to finally derive squared error.

➡️ **Next:** [Chapter 10 — Probability: Reasoning Under Uncertainty](<../Lecture 10 - Probability: Reasoning Under Uncertainty/blog.md>)
