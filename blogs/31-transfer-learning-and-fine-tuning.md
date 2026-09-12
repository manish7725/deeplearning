# Blog 31 — Can One Model Teach Another Model?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine a student who already knows multiplication.

Now you teach division.

You do not start by teaching what numbers are all over again.

You reuse existing knowledge.

Deep learning can do something similar.

A model trained on one large task can provide a useful starting point for another task.

This is **transfer learning**.

---

# 1. Start with a pretrained model

Suppose we train a large vision model on millions of images.

It may learn useful features:

```text
edges
 ↓
shapes
 ↓
parts
 ↓
objects
```

Now we want to recognize medical images.

Instead of starting from random weights, we can begin with the pretrained model.

---

# 2. Fine-tuning

We can update the pretrained model using our new dataset.

```text
pretrained model
       ↓
new task
       ↓
small amount of new training
       ↓
fine-tuned model
```

This is called **fine-tuning**.

The starting point already contains useful structure.

---

# 3. Freeze and train

Sometimes we freeze most of the network:

```text
pretrained layers → frozen
final layer        → train
```

The frozen layers keep their old knowledge.

The final layer learns the new task.

This can be useful when the new dataset is small.

---

# 4. Linear probing

Imagine the pretrained model produces a representation:

$$
z=f_\theta(x)
$$

We can keep $f_\theta$ fixed and train only a simple classifier:

$$
\hat y=Wz+b
$$

This is often called a **linear probe**.

It asks a useful question:

> “Is the information needed for my new task already present in this representation?”

MIT's transfer-learning material covers techniques including fine-tuning, linear probes, knowledge distillation and foundation models. citeturn1search7

---

# 5. Knowledge distillation

Imagine a very smart teacher helping a smaller student.

The large model is the **teacher**.

The smaller model is the **student**.

Instead of learning only from hard labels:

```text
cat = 1
```

the student can also learn the teacher's softer predictions:

```text
cat    0.80
dog    0.15
rabbit 0.05
```

That extra information can help the student learn relationships between classes.

---

# 6. Transfer can involve data too

Suppose your target dataset is tiny.

A related dataset can provide useful examples.

You can also use:

- augmentation
- synthetic data
- domain adaptation
- prompting

MIT separates transfer learning into model transfer and data transfer. citeturn1search11turn1search3

---

# 7. Foundation models

A foundation model is trained broadly and then adapted to many tasks.

Think:

```text
huge general training
        ↓
shared representation
   ┌────┼────┐
   ↓    ↓    ↓
task A task B task C
```

Large language models are a major example of this idea.

---

# 🧪 Think Like a Scientist

Train a small image classifier from scratch.

Then train another using a pretrained backbone.

Compare:

```text
training time
final accuracy
amount of data needed
```

Ask:

> Did the old knowledge make learning the new task easier?

---

# 🧠 What you should remember

1. Transfer learning reuses knowledge learned from another task or dataset.
2. Fine-tuning updates a pretrained model for a new task.
3. Linear probing keeps the representation fixed and trains a simple classifier.
4. Knowledge distillation transfers information from a larger teacher to a smaller student.
5. Data can also be transferred or adapted.
6. Foundation models are trained broadly so they can be reused across many tasks.

> **Training from zero is not always necessary. Sometimes the smartest first step is to reuse what another model already learned.**

---

# 🧪 Hands-on challenge

Take a pretrained image model.

Compare three experiments:

```text
A → train classifier from scratch
B → frozen pretrained backbone + new classifier
C → fine-tune pretrained backbone
```

Record accuracy and training time.

Explain why the three curves look different.