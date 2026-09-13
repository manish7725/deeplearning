# Lecture 10 — Probability: Reasoning Under Uncertainty

> **The Big Question:** What does it mean to put a number on something that has not happened yet?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2010%20-%20Probability%3A%20Reasoning%20Under%20Uncertainty/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Chapter 9 taught us to describe scatter — mean, standard deviation, correlation, and the bell shape that keeps appearing. All of it describes numbers we already have.
**Today:** We build the language for numbers we do *not* have yet, and discover that it obeys strict arithmetic rather than vague intuition.
**Next:** We will be able to compute the chance of evidence given a cause — and find we usually want the opposite.

---

## 1. The Problem: A Question About a Sale That Has Not Happened

The manager returns with the offer letter:

> *"A buyer has offered ₹14 lakh for the next flat. What are the chances it would have sold for more?"*

Everything Chapter 9 built is useless here, and it is worth seeing exactly why.

The mean is 12 and the standard deviation is 2, so ₹14 lakh sits one standard deviation above the middle. True, and not an answer — "one standard deviation above" is a *distance*, not a chance.

We could count. Of the ten recorded sales, exactly two exceeded 14:

$$
\frac{2}{10} = 20\%
$$

That feels like an answer. But look at what it actually says: *two of ten sales that already happened were above 14*. The manager is asking about a flat that has not been listed, to a buyer who has not yet decided. Those are different statements, and conflating them is the entire reason this chapter exists.

Push the counting approach a little and it breaks:

- **What is the chance the next flat sells for more than ₹20 lakh?** Zero of ten did, so the count says $0\%$ — "impossible". But nothing about this street makes ₹20 lakh impossible; we simply have not seen one yet.
- **What about exactly ₹11.5 lakh?** No sale hit that figure, so again $0\%$, even though it sits right in the middle of the data.
- **What if we had three sales instead of ten?** The counted answer would lurch wildly with each new record.

> 🧠 **Think** — Counting answers *"how often did this happen?"* We are asking *"how likely is this to happen?"* The first is a fact about the past. The second is a claim about the future, and we do not yet have a mathematics for it — we do not even have a definition.

---

## 2. What Would an Answer Need?

1. **A number, not a story.** "Fairly likely" cannot be computed with.
2. **A fixed scale**, so numbers from different questions can be compared and combined.
3. **Rules for combining.** If we know the chance of A and of B, we should be able to get the chance of "A or B", and of "A and B".
4. **Room for events never observed.** A ₹20 lakh sale must be allowed a small chance, not declared impossible.
5. **Agreement with counting when counting works.** For a fair coin the answer must be one half.

Requirement 3 is what turns this into mathematics rather than opinion, and requirement 4 is where simple counting fails hardest.

---

## 3. First Attempt: Relative Frequency

The honest starting point, because it satisfies requirements 1, 2 and 5. Run an experiment many times and define:

$$
P(A) \approx \frac{\text{number of times } A \text{ happened}}{\text{number of trials}}
$$

For a coin flipped 10,000 times and landing heads 5,013 times, $P(\text{heads}) \approx 0.5013$. That is convincing, and it satisfies requirement 5 exactly.

It fails requirement 4 completely, as §1 showed: anything not yet observed gets probability zero, which is not caution but a false claim of impossibility. And it cannot answer questions about one-off events at all — *"what is the chance this particular buyer walks away?"* has no sequence of trials to count.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Probability just means the fraction of times something happened."*
>
> That is a way to **estimate** a probability, not what one is. The frequency you measure wobbles with every new trial; the probability it estimates does not. Confusing the estimate with the thing estimated is the same error as confusing Chapter 9's sample mean $\bar x$ with the true mean $\mu$ — and Chapter 9 §9 showed those are genuinely different objects.

So we do what Chapter 6 did with the speedometer: stop trying to *measure* the quantity and instead write down the properties it must have.

---

## 4. The Discovery: Probability as a Measure

Start with the set of everything that could happen — the **sample space**, written $\Omega$. For one flat's sale price, $\Omega$ is every price it could fetch. For a die, $\Omega = \{1,2,3,4,5,6\}$.

An **event** is any subset of $\Omega$ — a collection of outcomes we care about. "Sells above ₹14 lakh" is an event; so is "rolls even".

A **probability** is a number assigned to each event, obeying three rules. These are not derived from anything; they are the definition, chosen because they are the minimum needed for requirements 1–4:

$$
\textbf{1.}\quad 0 \le P(A) \le 1
$$

$$
\textbf{2.}\quad P(\Omega) = 1
$$

$$
\textbf{3.}\quad P(A \text{ or } B) = P(A) + P(B) \quad\text{when } A,B \text{ cannot both happen}
$$

Rule 1 fixes the scale — 0 is impossible, 1 is certain. Rule 2 says something must happen. Rule 3 says probabilities of non-overlapping events add.

Everything else in probability follows from these three. Two immediate consequences worth deriving, because they are used constantly:

**The complement.** $A$ and "not $A$" cannot both happen, and together they are all of $\Omega$. So by rules 2 and 3:

$$
P(A) + P(\text{not } A) = 1
\qquad\Longrightarrow\qquad
\boxed{\;P(\text{not } A) = 1 - P(A)\;}
$$

Often the fastest route to a hard probability is the easy one of its opposite.

**Overlapping events.** Rule 3 needs $A$ and $B$ to be exclusive. If they can both happen, adding double-counts the overlap, so subtract it once:

$$
P(A \text{ or } B) = P(A) + P(B) - P(A \text{ and } B)
$$

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Probability is *area*. The sample space is a sheet of paper with area 1; an event is a region on it; the probability is how much of the sheet it covers. Overlapping regions must not be counted twice. |
| ✏️ **Numbers** | A die: $\Omega$ has six equally likely outcomes, so each covers $1/6$. "Even" covers three of them: $P = 3/6 = 1/2$. |
| 🎓 **Abstraction** | $P$ is a function from events to $[0,1]$ with $P(\Omega)=1$, additive over disjoint events. |

> 📜 **History Lens — A Gambler's Letter, 1654**
>
> Probability began with a dispute about how to split a pot. The Chevalier de Méré, a French writer and gambler, put a puzzle to **Blaise Pascal**: if a game of chance is interrupted before it finishes, how should the stakes be divided between players who are partway through?
>
> Pascal wrote to **Pierre de Fermat**, and their 1654 correspondence worked out that the fair split depends not on the score so far but on the *chances each player has of eventually winning* — reasoning about outcomes that never happened. That is the conceptual leap: assigning numbers to futures.
>
> It took another 279 years to put on a rigorous footing. In 1933 **Andrey Kolmogorov** published the three axioms in §4 above, defining probability as a measure on a set — the same mathematical machinery used for area and volume. That is why the intuition "probability is area" is not an analogy. It is the actual definition.
>
> Note what was needed. For nearly three centuries people computed probabilities correctly without agreeing what a probability *was* — exactly the situation Chapter 6 described for calculus between Newton and Weierstrass. **Useful mathematics often runs well ahead of its own foundations.**

---

## 5. Conditional Probability: When You Learn Something

Here is where probability stops being bookkeeping and starts being *reasoning*.

The office has 100 recent sales, classified two ways — whether the flat had 4 rooms, and whether it sold above ₹14 lakh:

| | sold > ₹14L | sold ≤ ₹14L | **total** |
|---|---:|---:|---:|
| **4 rooms** | 18 | 12 | **30** |
| **fewer rooms** | 7 | 63 | **70** |
| **total** | **25** | **75** | **100** |

Straight from the table, the chance a randomly chosen sale beat ₹14 lakh:

$$
P(>14) = \frac{25}{100} = 0.25
$$

Now the manager adds one fact: *this flat has 4 rooms.* Does the answer change?

It must. That fact eliminates 70 of the 100 sales from consideration — they are no longer possible outcomes. The sample space has **shrunk** to the 30 four-room sales, and within that smaller world, 18 beat ₹14 lakh:

$$
P(>14 \mid 4\text{ rooms}) = \frac{18}{30} = 0.6
$$

Read the bar as *"given that"*. Learning the flat has 4 rooms more than doubles the probability, from $0.25$ to $0.6$.

Generalize what we just did. We restricted to the $B$ outcomes, then asked what fraction of *those* were also $A$. In counts that is $\frac{\#(A \text{ and } B)}{\#B}$; divide top and bottom by the total to turn counts into probabilities:

$$
\boxed{\;P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}\;}
$$

Check it: $P(>14 \text{ and } 4\text{ rooms}) = 18/100 = 0.18$ and $P(4\text{ rooms}) = 0.30$, so $0.18/0.30 = 0.6$ ✓.

> 💡 **Intuition** — Conditioning is **zooming in**. You throw away the part of the sheet where $B$ is false, then rescale what remains so it again has area 1. Rescaling is the division by $P(B)$.

### The asymmetry that catches everyone

Using the same table, ask the *reverse* question — given that a flat sold above ₹14 lakh, what is the chance it had 4 rooms? Now restrict to the 25 high sales, of which 18 were four-room:

$$
P(4\text{ rooms} \mid >14) = \frac{18}{25} = 0.72
$$

$$
P(>14 \mid 4\text{ rooms}) = 0.6
\qquad\ne\qquad
P(4\text{ rooms} \mid >14) = 0.72
$$

**These are different questions with different answers**, and the words for them in English are nearly identical. Swapping them is among the most consequential errors in applied reasoning — Chapter 11 is built entirely around the machinery for converting one into the other, and around what goes wrong when people convert them by instinct instead.

---

## 6. Independence

Sometimes learning $B$ tells you nothing about $A$:

$$
P(A \mid B) = P(A)
$$

Then $A$ and $B$ are **independent**, and substituting into the conditional formula gives the version usually quoted:

$$
P(A \text{ and } B) = P(A)\,P(B) \qquad\textbf{(independent events only)}
$$

Two coin flips are independent; the first landing heads tells you nothing about the second. **Our rooms and price are not**: $P(>14 \mid 4\text{ rooms}) = 0.6 \ne 0.25 = P(>14)$, which is precisely why rooms is a useful feature for predicting price. Chapter 2 §9 measured the same relationship as a correlation of $0.70$.

> ⚠️ **Assuming independence is the most common silent error in probability.** It turns a hard calculation into an easy one, so the temptation is enormous. If two house prices on the same street are treated as independent, a market-wide crash — which moves all of them together — looks astronomically unlikely instead of merely unlikely. That specific mistake, applied to mortgage defaults, is a well-documented ingredient of the 2008 financial crisis.

---

## 7. Random Variables and Expectation

Events are yes/no. Prices are numbers, so we need a way to attach probabilities to numerical outcomes.

A **random variable** is a quantity whose value depends on the outcome — write it as a capital $X$. For a simplified street where a flat sells at one of three prices:

| price $x$ (₹ lakh) | 10 | 12 | 15 |
|---|---:|---:|---:|
| $P(X = x)$ | $0.5$ | $0.3$ | $0.2$ |

The probabilities sum to 1, as rule 2 demands. This table is the **distribution** of $X$.

Now: what is the *typical* price? Chapter 9 answered that for data by sharing the total equally. Here there is no data — only possibilities with weights. So weight each value by how likely it is:

$$
\mathbb{E}[X] = \sum_x x\,P(X = x) = 10(0.5) + 12(0.3) + 15(0.2) = 5 + 3.6 + 3 = \mathbf{11.6}
$$

This is the **expected value**, and it is the distribution's counterpart to Chapter 9's mean.

> ⚠️ **"Expected" is a bad name and you should distrust it.** The expected value here is ₹11.6 lakh, a price that **cannot occur** — the only possible prices are 10, 12 and 15. It is a balance point, not a prediction. Nobody should *expect* to see 11.6.

Spread works the same way. Chapter 9 averaged squared deviations; here we weight them:

$$
\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]
= 0.5(10-11.6)^2 + 0.3(12-11.6)^2 + 0.2(15-11.6)^2 = \mathbf{3.64}
$$

giving a standard deviation of $\sqrt{3.64} = 1.908$ ₹ lakh. And the shortcut from Chapter 9's Level-3 exercise reappears in this setting:

$$
\mathrm{Var}(X) = \mathbb{E}[X^2] - \big(\mathbb{E}[X]\big)^2 = 138.2 - 11.6^2 = 138.2 - 134.56 = 3.64 \;\checkmark
$$

### The distinction that Chapter 12 depends on

| | Chapter 9 | Chapter 10 |
|---|---|---|
| computed from | data you collected | a distribution you assumed |
| mean | $\bar x = \frac1n\sum x_i$ | $\mu = \mathbb{E}[X] = \sum x\,P(x)$ |
| spread | $s^2 = \frac{1}{n-1}\sum(x_i-\bar x)^2$ | $\sigma^2 = \mathbb{E}[(X-\mu)^2]$ |
| status | a **measurement** | a **property of the model** |

Keep these apart. $\bar x$ is what ten houses did; $\mu$ is what the process generating houses tends to do. **Chapter 12's entire argument is about the relationship between them** — it asks which $\mu$ makes the observed $\bar x$ most plausible.

---

## 8. 🔬 The Experiment: Do Frequencies Really Approach Probabilities?

§3 claimed relative frequency *estimates* probability. That claim deserves testing rather than assuming, since the whole chapter rests on it.

> 🧠 **Predict before reading on.** Simulate sales from the §7 distribution. After 10 sales, how close should the observed average be to $\mathbb{E}[X] = 11.6$? After 1,000? After 100,000? And does the *error* shrink in a pattern you already know?

Simulating and averaging, the running mean wanders at first and then settles onto 11.6. That is the **law of large numbers**: the sample mean of independent draws converges to the expected value.

But look at *how fast*. Chapter 9 §9's experiment already found the rule — the spread of a sample mean is $\sigma/\sqrt{n}$. With $\sigma = 1.908$:

| sales | typical error in the average |
|---:|---:|
| $10$ | $0.60$ |
| $100$ | $0.19$ |
| $10{,}000$ | $0.019$ |
| $1{,}000{,}000$ | $0.0019$ |

To get one extra decimal place of accuracy you need **a hundred times** the data. This is not a detail — it is the fundamental economics of learning from samples, and it explains why datasets get big: accuracy is expensive, and it gets more expensive the more you already have.

It also justifies §3 properly. Relative frequency *does* approach probability, and now we know the rate.

---

## 9. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **Zero from never observing** | "impossible" for anything unseen | §1. A count of 0 is not a probability of 0. |
| **Confusing $P(A\mid B)$ with $P(B\mid A)$** | confident, backwards conclusions | §5. $0.6 \ne 0.72$ on the same table. |
| **Assuming independence** | rare events look impossibly rare | §6. Correlated risks compound. |
| **Expecting the expected value** | planning for an impossible outcome | §7. $\mathbb{E}[X] = 11.6$ can never occur. |
| **Adding overlapping events** | totals above 1 | §4. Subtract the overlap once. |
| **Too few trials** | frequency wobbles wildly | §8. Error falls only as $1/\sqrt{n}$. |
| **Ignoring the base rate** | wildly overconfident diagnoses | Chapter 11's entire subject. |

---

## 10. 🎯 Machine Learning Connection

| This chapter | In machine learning |
|---|---|
| $P(A)$ on $[0,1]$ | what a classifier's output *means* (Ch 14) |
| distribution | the assumption that derives a loss (Ch 12) |
| $\mathbb{E}[X]$ | the target a regression model estimates |
| conditional probability | $P(\text{label} \mid \text{features})$ — what supervised learning computes |
| independence | the "naive" in naive Bayes (Ch 11) |
| law of large numbers | why more data helps, and why it helps so slowly |
| $\sigma/\sqrt{n}$ | why benchmark differences on small test sets mean nothing (Ch 21) |

The reframing worth carrying forward: **supervised learning is conditional probability**. A model given features $\mathbf{x}$ and asked for a label $y$ is being asked for $P(y \mid \mathbf{x})$ — exactly §5's operation, with $\mathbf{x}$ as the thing we condition on. Chapter 14's logistic regression outputs that number directly, and Chapter 15's softmax outputs a whole distribution over labels.

---

## 11. Distinctions That Matter

| | |
|---|---|
| **Frequency** — measured, wobbles | **Probability** — the thing being estimated |
| **Outcome** — one result | **Event** — a set of outcomes |
| $P(A \mid B)$ | $P(B \mid A)$ — a different question |
| **Independent** — $P(A\mid B) = P(A)$ | **Exclusive** — cannot both happen |
| $\bar x$ — mean of data (Ch 9) | $\mathbb{E}[X]$ — mean of a distribution |
| **Expected value** — a balance point | **A likely outcome** — may be neither |
| **Probability** — the model's claim | **Confidence** — often a claim about the modeller |

Note especially that *independent* and *exclusive* are near-opposites, despite sounding similar. If $A$ and $B$ are exclusive, learning $A$ happened tells you $B$ definitely did not — which is a very strong dependence.

---

## 12. What We Discovered

1. Chapter 9's tools are backward-looking; "what are the chances?" is a different kind of question.
2. Counting past frequencies assigns zero to everything unseen, which claims impossibility rather than ignorance.
3. Rather than measuring probability, we state the properties it must have — Kolmogorov's three axioms.
4. The complement rule and the overlap rule both follow from those three lines.
5. Conditioning is zooming in: discard outcomes where $B$ is false, rescale the rest by dividing by $P(B)$.
6. $P(A\mid B)$ and $P(B\mid A)$ are different questions — $0.6$ versus $0.72$ on the very same table.
7. Independence means conditioning changes nothing; assuming it wrongly makes correlated disasters look impossible.
8. Expected value weights outcomes by probability, and may be a value that can never occur.
9. Frequencies do converge to probabilities, but the error falls only as $1/\sqrt{n}$ — a hundredfold more data per extra decimal place.

## 13. Mathematics We Built

$$
0 \le P(A) \le 1
\qquad
P(\Omega) = 1
\qquad
P(A \cup B) = P(A) + P(B) \;\text{ if disjoint}
$$

$$
P(\text{not }A) = 1 - P(A)
\qquad
P(A \cup B) = P(A) + P(B) - P(A\cap B)
$$

$$
P(A\mid B) = \frac{P(A \cap B)}{P(B)}
\qquad
P(A\cap B) = P(A)P(B) \;\text{ if independent}
$$

$$
\mathbb{E}[X] = \sum_x x\,P(X=x)
\qquad
\mathrm{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

## 14. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $\Omega$ | "omega" — everything that could happen | the set of outcomes |
| $P(A)$ | the probability of event $A$ | `p_a` |
| $P(A \mid B)$ | "probability of A **given** B" | `p_a_given_b` |
| $A \cap B$ | both happen | `a and b` |
| $A \cup B$ | at least one happens | `a or b` |
| $X$ | a random variable — a number that depends on chance | `X` |
| $\mathbb{E}[X]$ | expected value — probability-weighted mean | `(values * probs).sum()` |
| $\mathrm{Var}(X)$ | spread of a distribution | — |
| $\mu, \sigma$ | true mean and standard deviation | — |

## 15. One-Minute Explanation

With no equations:

> Two of our ten flats sold above ₹14 lakh. Why is "20%" not a correct answer to "what are the chances the next one does?" — and what would a correct answer even be a statement about?

---

## 16. Exercises

**Level 1 — Observe.** From §5's table, read off: how many sales were there in total; how many four-room flats sold at or below ₹14 lakh; and what fraction of *all* sales were four-room flats above ₹14 lakh. Then explain in one sentence why $P(>14 \mid 4\text{ rooms})$ and $P(4\text{ rooms} \mid >14)$ have the same numerator but different answers.

**Level 2 — Calculate (by hand, no code).** A die is rolled once. Compute $P(\text{even})$, $P(>4)$, $P(\text{even and} >4)$ and $P(\text{even or} >4)$ — the last one using the overlap rule, then verify by counting outcomes directly. Then check whether "even" and "$>4$" are independent by comparing $P(\text{even}\mid >4)$ with $P(\text{even})$.

**Level 3 — Derive.** From the three axioms alone, prove $P(\text{not }A) = 1-P(A)$ and $P(A\cup B) = P(A)+P(B)-P(A\cap B)$. Then prove $\mathrm{Var}(X) = \mathbb{E}[X^2]-\mathbb{E}[X]^2$ by expanding $\mathbb{E}[(X-\mu)^2]$, stating clearly where you use the fact that $\mathbb{E}$ is a weighted sum. Finally: show that if $A$ and $B$ are exclusive *and* both have non-zero probability, they cannot be independent.

**Level 4 — Investigate** (notebook Steps 8–11). Simulate sales from §7's distribution and track the running average. Measure how the error from $11.6$ shrinks as $n$ grows, and confirm it follows $\sigma/\sqrt{n}$. Then change the distribution so one outcome is very rare (probability $0.001$) and repeat: how many samples before the running mean is reliable, and what does that say about learning from imbalanced data?

**Level 5 — Design.** The office wants a probability for *"this flat sells above ₹20 lakh"* — a price no recorded sale has reached, so counting returns 0. Design a method that assigns it a small positive probability instead. State the assumption your method makes, compute a number, and then say honestly what would make your assumption wrong. (You are inventing the idea behind smoothing, which Chapter 11's naive Bayes needs.)

---

## 17. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "Probability is the fraction of times it happened." | That is an estimate of it, and it wobbles. §3 |
| "It has never happened, so the probability is zero." | Zero means *impossible*, which is a far stronger claim than *unobserved*. §1 |
| "$P(A\mid B)$ is basically $P(B\mid A)$." | $0.6$ versus $0.72$ on one table. §5 |
| "Independent means they cannot both happen." | That is *exclusive*, which is a strong dependence. §11 |
| "The expected value is what I should expect." | $\mathbb{E}[X]=11.6$ when only 10, 12 and 15 are possible. §7 |
| "10,000 samples is plenty for any question." | Depends entirely on the rarity of the event. §8 |
| "I can just multiply the probabilities." | Only if independent — and that is usually assumed, not checked. §6 |

## 18. Socratic Questions

1. Kolmogorov's axioms never say what a probability *is* — only how it behaves. Is that a gap, or is it the point?
2. §7's expected value is a price that cannot occur. What does that tell you about a regression model trained to predict the mean of a bimodal distribution?
3. If $P(A\mid B) = P(A)$, does it follow that $P(B\mid A) = P(B)$? Prove it or find a counterexample.
4. The error in an estimated probability falls as $1/\sqrt{n}$. If a rival model beats yours by 0.3% on a 1,000-item test set, should you believe it?
5. We said supervised learning computes $P(y\mid\mathbf{x})$. What is the sample space, and what exactly is being conditioned on?
6. Our ten sales gave a 20% frequency for "above 14". If the true probability were 25%, how surprised should we be to have seen exactly 2 out of 10?

---

## 19. 🔭 Bridge to Chapter 11

We can now put numbers on futures, combine them with rules, and update them when we learn something — §5's conditioning is genuinely a machine for reasoning, not just bookkeeping.

But notice which direction that machine runs.

Every conditional probability we computed went **from cause to evidence**. Given that a flat has 4 rooms, what is the chance it sells high? Given that a coin is fair, what is the chance of six heads? In each case we assumed the situation and computed the chance of the observation.

Real questions almost always run the other way. A surveyor's damp-detector alarms — what is the chance the house *actually has damp*? We know $P(\text{alarm} \mid \text{damp})$, because that is what the manufacturer tested. We want $P(\text{damp} \mid \text{alarm})$, which is what the buyer is paying to know.

§5 already warned that these are different numbers — $0.6$ and $0.72$ on the same table. So we cannot simply swap them. And the stakes are higher than they look: if a detector is advertised as "95% accurate" and it alarms on your house, most people — including most professionals — will conclude the house is about 95% likely to have damp.

That conclusion can be off by a factor of six, and the reason is a number nobody thought to ask for.

> **We can compute the chance of the evidence given the cause. How do we turn that around and get the chance of the cause given the evidence?**

That is where Chapter 11 begins. Its answer is a single line of algebra with consequences so counterintuitive that it reshapes how you read every test result, every diagnostic, and — in Chapter 12 — every loss function in this course.

➡️ **Next:** [Chapter 11 — Bayes' Rule: What Evidence Does to Belief](<../Lecture 11 - Bayes' Rule: What Evidence Does to Belief/blog.md>)
