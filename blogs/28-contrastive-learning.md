# Blog 28 — How Can a Machine Learn What Is Similar?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine you have two photographs of the same dog.

One is bright.

One is dark.

You want a machine to understand:

> “These are still the same kind of thing.”

Now show it a dog and a bicycle.

You want their representations to be far apart.

This idea leads to **similarity-based representation learning** and **contrastive learning**.

---

# 1. Put similar things close together

Imagine every object becomes a point on a map.

```text
       dogs 🐕 🐕
          ● ●
        ●

                         ● 🚲
                         bicycle
```

A good representation might place similar examples near one another.

Mathematically, we can use a similarity score such as a dot product:

$$
s(a,b)=a^Tb
$$

A larger value can mean the vectors point in more similar directions.

---

# 2. Positive and negative pairs

Suppose we have:

```text
image A = dog
image B = same dog, different crop
image C = bicycle
```

Then:

```text
A-B → positive pair
A-C → negative pair
```

The learning objective says:

> Pull positive pairs closer. Push negative pairs apart.

---

# 3. Why this is clever

What if nobody gives us a label saying “dog”?

We can create two views of the same image ourselves:

```text
original image
     ↓
 ┌───┴────┐
 ↓        ↓
 crop    color change
 ↓        ↓
view A   view B
```

We know A and B came from the same image.

That gives us a learning signal without a human-written class label.

This is one reason contrastive learning is important for **self-supervised learning**.

MIT's similarity-based representation lecture covers metric learning, contrastive learning, self-supervised learning, InfoNCE, alignment and uniformity, and hard negatives. citeturn1search9

---

# 4. A simple contrastive loss

A common form is called **InfoNCE**.

One simplified view is:

$$
L=-\log\frac{e^{s(x,y)/\tau}}{\sum_j e^{s(x,y_j)/\tau}}
$$

Do not memorize it yet.

Read it as a story:

```text
score the correct pair
        ↓
compare it with other pairs
        ↓
reward the correct pair
        ↓
penalize confusing pairs
```

The temperature $\tau$ controls how strongly the model focuses on differences in similarity.

---

# 5. Hard negatives

Suppose the model sees:

```text
positive → dog
negative 1 → bicycle
negative 2 → cat
negative 3 → another dog
```

The last example may be much harder.

It looks very similar but is actually a negative example for the current pair.

Such examples are called **hard negatives**.

They can teach the model finer distinctions.

---

# 🧠 A school example

Imagine sorting students by favorite sport.

Two students who both love cricket might be close on a “sports preference map.”

Someone who loves chess might be farther away.

The machine is not learning the word “cricket” directly.

It is learning a geometry where relationships become useful.

---

# 🧪 Think Like a Scientist

Take a dataset of images.

Create two random augmentations of every image.

Train an encoder so that two views of the same image have similar embeddings.

Then visualize the embeddings with a 2D projection.

Ask:

> Do images that look similar begin to form groups?

If yes, you can literally see representation learning happening.

---

# 🧠 What you should remember

1. Representations can be organized by similarity.
2. Positive pairs should become similar.
3. Negative pairs should become different.
4. Contrastive learning can learn without traditional labels.
5. Augmentations can create useful positive pairs.
6. Hard negatives can teach finer distinctions.

> **Instead of telling the machine what every object is, sometimes we can teach it which things belong together.**

---

# 🧪 Hands-on challenge

Create pairs of images:

```text
same image → positive
same class → positive
random different class → negative
```

Train a small encoder.

Measure the average distance between positive and negative pairs before and after training.

Your experiment should answer:

> **Did the representation learn a more useful geometry?**