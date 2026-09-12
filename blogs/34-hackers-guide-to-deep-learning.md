# Blog 34 — The Detective's Guide to Training a Neural Network

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine your science experiment suddenly stops working.

The graph is flat.

The loss is `NaN`.

Accuracy is stuck at 10%.

What do you do?

You become a detective. 🔎

Training a neural network is partly mathematics and partly careful experimentation.

---

# 1. First question: does the data make sense?

Before blaming the neural network, inspect the data.

Ask:

```text
Are labels correct?
Are values in the expected range?
Are there missing values?
Are images loaded correctly?
Are train and test data separated?
```

A perfect algorithm cannot rescue completely broken data.

---

# 2. Can the model learn one tiny example?

This is one of the most useful debugging experiments.

Take one example.

Try to make the model memorize it.

```text
one input
   ↓
model
   ↓
train repeatedly
   ↓
loss should become very small
```

If it cannot memorize one simple example, something fundamental may be wrong.

Possible causes:

- incorrect labels
- wrong loss
- broken forward pass
- optimizer problem
- learning rate problem
- tensor-shape error

---

# 3. Look at the loss curve

A healthy training experiment often shows some movement in loss.

```text
loss
│●
│  ●
│    ●
│      ●
│        ●
└──────────── step
```

If it is completely flat, ask why.

If it explodes:

```text
1
2
5
20
1000
NaN
```

your updates may be too large or the computation may be numerically unstable.

---

# 4. Learning rate detective

Try:

```text
0.1
0.01
0.001
0.0001
```

You are performing a controlled experiment.

Do not change ten things at once.

Change one important variable and observe the result.

---

# 5. Check shapes

Many neural-network bugs are really shape bugs.

Suppose:

$$
X\in\mathbb R^{32\times 10}
$$

means:

```text
32 examples
10 features each
```

and

$$
W\in\mathbb R^{10\times 4}
$$

Then:

$$
XW\in\mathbb R^{32\times4}
$$

Write the shapes beside your equations.

It is one of the simplest debugging tools in deep learning.

---

# 6. Train and validation curves

Suppose:

```text
training loss   ↓↓↓
validation loss  ↓ then ↑
```

That can be a clue that the model is beginning to overfit.

Plots are not decoration.

They are evidence.

---

# 7. Keep an experiment notebook

For every experiment record:

| Item | Value |
|---|---|
| model | CNN-small |
| learning rate | 0.001 |
| batch size | 64 |
| epochs | 20 |
| train loss | ... |
| validation accuracy | ... |

Then you can compare experiments instead of relying on memory.

---

# 🧠 The scientific loop

```text
Question
   ↓
Hypothesis
   ↓
Experiment
   ↓
Measurement
   ↓
Observation
   ↓
New hypothesis
```

This is exactly how you should approach ML debugging.

MIT includes a dedicated practical “Hacker's Guide to Deep Learning” session alongside its theory lectures. citeturn1search1turn1search3

---

# 🧪 Think Like a Scientist

Create a model that deliberately fails.

Examples:

- learning rate too high
- labels shuffled
- wrong output size
- input normalization removed

Then diagnose each failure using only:

```text
loss curve
accuracy
tensor shapes
small-data experiment
```

This teaches more than simply making a model that works.

---

# 🧠 What you should remember

1. Debug the data before blaming the model.
2. Try to overfit one tiny example.
3. Plot training and validation curves.
4. Check tensor shapes.
5. Change one important variable at a time.
6. Keep an experiment log.
7. Treat ML experiments like science experiments.

> **A strong ML engineer is not the person whose models never fail. It is the person who can explain why a model failed.**