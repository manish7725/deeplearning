# Blog 36 — How Do We Measure What a Neural Network Has Learned?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine a teacher says:

> “This student understands mathematics.”

How do we know?

We need evidence.

Maybe the student can solve new problems.

Maybe they can explain the method.

Maybe they can solve harder problems.

Deep learning needs the same kind of thinking.

A model's final accuracy is useful, but sometimes we want to understand **what the model learned internally**.

---

# 1. Measure the output

The simplest measurement is task performance.

For classification:

$$
\text{accuracy}=\frac{\text{correct predictions}}{\text{total predictions}}
$$

For regression we might use mean squared error.

These tell us whether the model is useful for a particular task.

---

# 2. Measure the representation

Suppose an encoder produces:

$$
z=f(x)
$$

We can ask:

> Do similar examples have similar representations?

For example:

```text
cat → [0.2, 0.8, ...]
cat → [0.3, 0.7, ...]
dog → [0.9, 0.1, ...]
```

We can measure distances or similarities between these vectors.

---

# 3. Visualize embeddings

A representation may have hundreds of dimensions.

Humans cannot easily draw that.

So we can use a dimensionality-reduction method to create a 2D visualization.

Then we may see:

```text
● ● ●       ▲ ▲
 ● ●       ▲ ▲ ▲

cats       dogs
```

The picture is only a visualization, not proof that the representation is perfect.

It is a tool for asking questions.

---

# 4. Probe the representation

Suppose we freeze a pretrained model.

Then train a small classifier on its representation.

If the small classifier can easily predict a property, that property may already be encoded in the representation.

This connects to **linear probing** from transfer learning.

---

# 5. Measure behavior, not just numbers

Imagine two models both have 95% accuracy.

Model A fails on night images.

Model B fails on rotated images.

Their single accuracy score hides an important difference.

So we should measure performance across conditions:

```text
normal
night
rotation
new background
new device
```

This connects measurement with robustness and OOD testing.

---

# 6. Why this is useful

Measurement helps us answer questions such as:

- Did the model learn useful features?
- Is the representation reusable?
- Where does the model fail?
- Did a new training method improve anything?
- Is a larger model actually better?

MIT includes a dedicated session on metrized deep learning, reflecting the idea that measuring learned systems can itself be a deep-learning research problem. citeturn1search0

---

# 🧪 Think Like a Scientist

Train two models.

Do not compare only final accuracy.

Measure:

```text
accuracy
loss
parameter count
training time
OOD accuracy
embedding similarity
```

Now write a conclusion.

You may discover that the “best” model depends on what you measure.

---

# 🧠 What you should remember

1. A model should be measured using the task it is meant to solve.
2. Accuracy alone can hide important behavior.
3. Representations can also be studied.
4. Probing can test whether useful information is present.
5. Evaluation should include realistic conditions.
6. Good experiments measure more than one number.

> **If you cannot measure what changed, it is difficult to know whether your new idea actually helped.**

---

# 🧪 Hands-on challenge

Take two models with similar accuracy.

Create a small evaluation report containing:

```text
accuracy
validation loss
OOD accuracy
training time
model size
```

Then answer:

> Which model would you deploy, and why?

Your answer must use evidence, not just the largest accuracy number.