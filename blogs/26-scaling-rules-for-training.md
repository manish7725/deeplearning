# Blog 26 — What Happens When We Make Training Bigger?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine pushing a small toy car.

A tiny push moves it.

Now imagine pushing a huge truck with the same tiny push.

The result is different.

Training neural networks has a similar problem.

When we change the size of a model or the amount of data processed at once, our training settings may need to change too.

This is the idea behind **scaling rules for optimization**.

---

# 1. The learning rate is a step size

Gradient descent says:

$$
\theta_{new}=\theta-\eta\nabla L
$$

Here $\eta$ is the learning rate.

Think of it as the size of your step downhill.

```text
small step → slow but careful
large step → fast but may jump around
```

---

# 2. Batch size

Suppose we have 10,000 examples.

We could calculate a gradient using:

```text
1 example
10 examples
100 examples
1000 examples
```

The group used for one update is called a **batch**.

A larger batch usually gives a less noisy estimate of the gradient.

A smaller batch gives a noisier estimate but can be cheaper per update.

---

# 3. Why batch size changes training

Imagine asking one student:

> “Which route should we take?”

You get one opinion.

Ask 100 students.

The average answer may be more stable.

A batch works similarly:

```text
examples
  ↓
calculate many gradients
  ↓
average
  ↓
update weights
```

---

# 4. A simple experiment

Train the same model with:

```text
batch = 8
batch = 32
batch = 128
```

Record:

- training time
- number of updates
- final validation loss

You may discover that changing batch size changes the practical behavior of training.

---

# 5. Bigger does not automatically mean better

Suppose model A has:

```text
1 million parameters
```

and model B has:

```text
100 million parameters
```

Model B has more capacity.

But it also needs appropriate training.

You may need:

- more data
- more compute
- suitable learning rates
- suitable initialization
- enough training steps

MIT's course treats optimization scaling separately from scaling laws because the behavior of training itself changes as model and batch sizes change. citeturn0search10turn0search0

---

# 6. The deeper idea

There are several things we can scale:

```text
Model size
Data size
Batch size
Training steps
Compute
```

They are connected.

Changing one may change what works for the others.

This is why training a giant neural network is not simply:

> “Run the small experiment for longer.”

It becomes an engineering and mathematical problem.

---

# 🧠 What you should remember

1. Learning rate controls update size.
2. Batch size controls how many examples help make one update.
3. Larger models usually require more computation and often more data.
4. Training settings interact with model size.
5. Scaling a neural network is a system problem, not just adding more neurons.

> **When the learner becomes bigger, the rules for teaching it may need to change too.**

---

# 🧪 Hands-on challenge

Train one model three times with different batch sizes.

Keep everything else fixed.

Plot:

$$
\text{loss}\quad\text{vs}\quad\text{training step}
$$

Then repeat with two learning rates.

Describe what changes and what stays the same.

You are now experimenting with optimization scaling.