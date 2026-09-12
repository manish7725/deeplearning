# Blog 30 — What Happens When the World Changes?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine you learn to identify bicycles from photographs taken on sunny days.

Then someone gives your model a bicycle photograph taken at night.

The bicycle did not change.

The world around it did.

Will the model still work?

This is the problem of **distribution shift** and **out-of-distribution (OOD) generalization**.

---

# 1. Training world vs real world

During training:

```text
bright images
clear backgrounds
normal camera
```

During deployment:

```text
dark images
new backgrounds
new camera
```

The training distribution and deployment distribution are no longer identical.

---

# 2. Distribution

A distribution describes how likely different examples are.

You do not need advanced probability to understand the basic idea.

Imagine a bag containing mostly red balls and a few blue balls.

The distribution tells us how frequently different colors appear.

A machine-learning model learns from one distribution.

It is often evaluated on another sample from a similar distribution.

But sometimes the new world is different.

---

# 3. Covariate shift

Suppose the input changes:

```text
training: mostly daytime images
new world: mostly nighttime images
```

The input distribution changed.

This is one form of distribution shift.

Other shifts can happen too.

For example:

```text
labels change
relationships change
new environments appear
```

---

# 4. Shortcut learning

Imagine every training picture of a cow has green grass behind it.

The model may accidentally learn:

> “Green grass → cow.”

Then you show a cow standing on a road.

The model may become confused.

It learned a **shortcut** rather than the concept we wanted.

This is why good test accuracy is not always enough.

MIT's OOD material discusses distribution shifts, adversarial robustness and shortcut-like failures in learned models. citeturn0search5turn1search3

---

# 5. Adversarial examples

Sometimes a tiny, carefully chosen change can fool a model.

Humans may see:

```text
cat
```

while a model sees:

```text
something else
```

The important lesson is not that computers are “stupid.”

It is that the model's decision boundary may behave differently from human intuition.

---

# 6. Robustness

A robust model should continue to work when irrelevant details change.

Think:

```text
rain
night
small camera change
slight noise
new background
```

Robustness asks:

> **Does the model focus on the right information?**

---

# 🧪 Think Like a Scientist

Train a classifier on centered objects with plain backgrounds.

Then create a test set with:

- different backgrounds
- different brightness
- small rotations
- different object positions

Measure accuracy on every group.

You may discover:

```text
normal test → 95%
new background → 72%
night images → 61%
```

Now you know exactly where your model fails.

---

# 🧠 What you should remember

1. The real world may differ from the training world.
2. Distribution shift can hurt accuracy.
3. Models can learn shortcuts.
4. OOD generalization asks whether a model works outside familiar conditions.
5. Robustness means being less sensitive to irrelevant changes.
6. Testing one clean dataset is not enough for a production model.

> **A model is not truly useful because it works in the classroom. It is useful when it survives the messy world outside the classroom.**

---

# 🧪 Hands-on challenge

Create three test sets:

```text
A → same distribution
B → new background
C → new lighting
```

Train only on A.

Evaluate on A, B and C.

Plot accuracy.

Then add augmented training data and repeat.

Did robustness improve?