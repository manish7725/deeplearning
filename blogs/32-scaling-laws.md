# Blog 32 — Why Do Bigger Models Often Learn More?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine studying for an exam.

You have:

```text
10 practice questions
```

Then your teacher gives you:

```text
10,000 practice questions
```

More practice may help.

Now imagine also having a much better textbook and more study time.

Modern AI has a similar story.

We can scale:

```text
model size
training data
compute
```

Researchers have found regular patterns in how performance changes with these quantities.

These are called **scaling laws**.

---

# 1. What is scaling?

A model has parameters.

A small model might have:

```text
1 million
```

A larger one might have:

```text
100 million
```

or much more.

We can also increase the amount of training data.

And we can increase the amount of computation used during training.

---

# 2. More is not enough

Imagine three students:

```text
Student A → huge textbook, no study
Student B → tiny textbook, perfect study
Student C → huge textbook + lots of study
```

The biggest textbook alone does not guarantee success.

Likewise:

> **Model size, data and compute interact.**

---

# 3. A simple mathematical pattern

A scaling relationship is often approximated by a power law:

$$
L(N)\approx aN^{-b}+c
$$

where $N$ might represent model size and $L$ might represent loss.

The exact constants depend on the experiment.

Do not focus on memorizing the equation.

Focus on the shape:

```text
bigger model
     ↓
usually lower loss
     ↓
but improvement becomes harder
```

MIT's course studies empirical scaling laws for neural language models and compute-optimal training. citeturn0search8turn1search3

---

# 4. The surprising part

Suppose doubling model size improves performance.

You might think:

> “Then keep doubling forever.”

But training a model costs computation.

Eventually the question becomes:

> **Where should I spend my limited compute?**

Should I use:

```text
bigger model?
more data?
more training steps?
```

This is the idea of **compute-optimal training**.

---

# 5. Why scaling laws matter for LLMs

Large language models are trained by predicting tokens.

The training process can involve enormous datasets and computation.

If researchers can estimate how performance changes with model size, data and compute, they can plan experiments more intelligently.

Instead of randomly guessing:

```text
model = huge
train = huge
hope = yes
```

we can make a more informed plan.

---

# 🧠 A school analogy

Suppose you have 100 hours to study.

You could spend:

```text
90 hours reading
10 hours practicing
```

or:

```text
50 hours reading
50 hours practicing
```

The best allocation depends on the subject.

Scaling is similar: resources must be allocated intelligently.

---

# 🧪 Think Like a Scientist

Train models with:

```text
10K parameters
100K parameters
1M parameters
```

Keep the dataset type similar.

Record final loss.

Plot:

$$
\text{loss}\quad\text{vs}\quad\text{model size}
$$

Does the curve look roughly smooth?

You have performed a tiny scaling-law experiment.

---

# 🧠 What you should remember

1. Deep-learning systems can be scaled in model size, data and compute.
2. Performance often changes in predictable ways over useful ranges.
3. Larger models can improve performance but cost more computation.
4. Data and model size need to be considered together.
5. Scaling laws help researchers plan large training runs.
6. Compute-optimal training asks how to use a limited compute budget wisely.

> **Modern AI is not only about inventing a better model. It is also about understanding how size, data and computation work together.**

---

# 🧪 Hands-on challenge

Train at least three model sizes.

Create a table:

| Model size | Training compute | Final loss |
|---:|---:|---:|
| small | ... | ... |
| medium | ... | ... |
| large | ... | ... |

Plot the results.

Then ask:

> Was the largest model also the most efficient use of compute?

That question is the beginning of scaling science.