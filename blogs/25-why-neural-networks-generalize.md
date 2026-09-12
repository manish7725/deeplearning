# Blog 25 — Why Does a Neural Network Work on New Examples?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine you memorize every answer in a school workbook.

Then your teacher gives you a slightly different problem.

If you only memorized, you may fail.

If you understood the idea, you can solve it.

Machine learning has the same challenge.

A model must not merely remember its training examples.

It should learn patterns that work on **new examples**.

This is called **generalization**.

---

# 1. Training is not the final exam

Suppose we train on:

```text
1000 training pictures
```

The model gets 99% correct.

Wonderful?

Not necessarily.

Now give it:

```text
1000 new pictures
```

If it gets only 60% correct, the model memorized too much and learned too little.

So we care about two errors:

$$
\text{training error}
$$

and

$$
\text{test error}
$$

The difference is part of the **generalization gap**.

---

# 2. The strange thing about big networks

Modern neural networks can have enormous numbers of parameters.

You might think:

> “Surely a huge model must memorize everything.”

Sometimes it can memorize training data extremely well.

Yet large neural networks can also perform very well on unseen data.

That is one of the surprising questions in modern deep-learning theory.

MIT's course explicitly studies why neural networks generalize, including generalization in high dimensions and the behavior of highly expressive networks. citeturn0search6turn0search9

---

# 3. A simple example

Imagine learning to recognize triangles.

Training examples:

```text
🔺  🔺  🔺  🔺
```

A bad learner might remember:

> “A triangle is this exact picture.”

A better learner notices:

> “A triangle has three sides.”

The second rule works when:

- the triangle is larger
- the triangle is smaller
- the triangle is rotated
- the color changes

The model learned a useful pattern rather than memorizing pixels.

---

# 4. Inductive bias

How does a model decide which patterns are useful?

Partly through its **inductive bias**.

That means the assumptions built into the learning system.

For example:

```text
CNN
→ nearby pixels matter

GNN
→ relationships matter

Transformer
→ interactions between tokens matter
```

The architecture itself gives the learner a starting idea about the world.

---

# 5. Overfitting

Imagine drawing a curve through every training point:

```text
●────●────●────●
 \  / \  / \  /
  \/   \/   \/
```

It may pass through every example perfectly.

But it may behave badly between the examples.

This is **overfitting**.

The model has become too focused on the training data.

A simpler pattern might be better.

---

# 6. Bias and variance — a first look

Think about throwing darts.

```text
High bias:
all darts far from the center

High variance:
darts scattered everywhere

Good model:
darts close to the center
```

Very roughly:

- **bias** → model is too simple or makes systematic mistakes
- **variance** → model reacts too strongly to the particular training sample

Deep learning complicates this old picture because modern networks can be highly overparameterized and still generalize well.

So the simple school-level picture is useful, but it is not the whole story.

---

# 7. Double descent

There is an especially surprising modern phenomenon.

As model complexity increases, test error does not always behave like a simple U-shaped curve.

In some settings it can decrease again after a peak.

A cartoon might look like:

```text
error
  |
  | \      /
  |  \____/ \____
  |______________ complexity
```

This is often called **double descent**.

You do not need the full theory yet.

The important lesson is:

> **Modern deep learning does not always behave like the simplest textbook picture of overfitting.**

---

# 8. Train, validate, test

A practical workflow is:

```text
Dataset
  ↓
Train set
  ↓
learn parameters

Validation set
  ↓
choose settings

Test set
  ↓
final evaluation
```

The test set should behave like a final exam.

If you repeatedly tune your model against the test set, it stops being a fair final exam.

---

# 🧪 Think Like a Scientist

Train a model on 20 examples.

Then train another on 2,000 examples.

Keep the architecture fixed.

Compare:

```text
training accuracy
validation accuracy
```

Ask:

> What changed when the model saw more examples?

This is a real experiment about generalization.

---

# 🧠 What you should remember

1. Training accuracy is not enough.
2. Generalization means performing well on unseen examples.
3. Overfitting means learning the training set too specifically.
4. Architecture creates inductive bias.
5. Large networks can memorize and still generalize.
6. Modern deep learning has phenomena such as double descent that go beyond the simplest bias-variance story.

> **Learning is not remembering the examples. Learning is discovering a pattern that survives new examples.**

---

# 🧪 Hands-on challenge

Create a tiny classification dataset.

Train the same model with:

```text
10 examples
100 examples
1000 examples
```

Keep the test set unchanged.

Plot test accuracy against the amount of training data.

Then write three sentences explaining what happened.

That is your first real experiment in **learning theory**.