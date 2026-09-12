# Blog 23 — Why Should a Neural Network Care If We Move the Object?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine a photograph of a cat.

Now slide the cat ten centimetres to the right.

Did the animal stop being a cat?

Of course not. 🐈

But a computer sees numbers arranged in different positions.

So deep learning has a beautiful question:

> **How can a model learn that some changes should not change the answer?**

The answer takes us into **geometry, invariance and equivariance**.

---

# 1. A transformation changes the picture

Imagine a point:

$$
\mathbf x=\begin{bmatrix}2\\1\end{bmatrix}
$$

Move it two units right:

$$
\mathbf x'=\begin{bmatrix}4\\1\end{bmatrix}
$$

The coordinates changed.

The object may not have changed in the way we care about.

Mathematics gives us a way to describe such changes using transformations.

---

# 2. Invariance means “the answer stays the same”

Suppose a classifier is:

$$
f(x)=\text{cat}
$$

If a transformation $T$ moves the cat, we would like:

$$
f(T(x))=f(x)
$$

That is **invariance**.

The input changes.

The important answer does not.

Examples:

```text
move cat       → still cat
small resize   → still cat
slight rotation → still cat
```

Not every transformation should be ignored. A huge transformation can change what the object actually is or make it impossible to recognize.

So invariance is about choosing the changes that should not matter for a task.

---

# 3. Equivariance means “the output changes predictably”

Suppose an image contains an edge.

If we move the image right, the edge should move right too.

We might want:

$$
f(T(x))=T(f(x))
$$

The output follows the same transformation.

That is **equivariance**.

Think of a school map.

If you move the whole map five centimetres right, the location of the school on the map also moves five centimetres right.

The representation changes in a predictable way.

---

# 4. Why CNNs are useful

Convolution is naturally connected to local patterns.

A small filter can look for something like an edge:

```text
⬜⬜⬛
⬜⬛⬛
⬛⬛⬛
```

If the same edge appears somewhere else, the same filter can detect it there too.

That means a CNN does not need a completely different detector for every possible position.

This is an example of **shared structure**.

The architecture contains an assumption:

> “The same local pattern may matter wherever it appears.”

That assumption is called an **inductive bias**.

---

# 5. Architecture is a choice about the world

Different data has different structure.

```text
Image       → local neighborhoods
Sequence    → order
Graph       → connections
Set         → elements without order
```

A good architecture uses the structure that already exists in the data.

This is one of the deepest ideas in modern deep learning:

> **Architecture is not just a collection of layers. It is a guess about the structure of the world.**

---

# 🧠 A Class 7 example

Suppose you recognize your school from a photograph.

If the photograph is moved slightly left on your screen, should your answer change?

No.

That suggests some translation invariance.

Now imagine a weather map.

If a storm moves east, the location of the storm should move east in the representation.

That is closer to equivariance.

Same word, different idea:

```text
INVARIANCE
input changes
answer stays same

EQUIVARIANCE
input changes
output changes predictably
```

---

# 🧪 Think Like a Scientist

Take one image.

Create:

1. original image
2. shifted image
3. rotated image
4. resized image

Ask:

> Which transformations should leave the label unchanged?

Then ask:

> Which internal features should move when the object moves?

You have just started thinking about the geometry of deep learning.

---

# 🧠 What you should remember

1. Transformations change representations.
2. Invariance means the task output remains the same under a chosen transformation.
3. Equivariance means the output changes predictably with the input.
4. CNNs exploit local structure and shared filters.
5. Architecture contains assumptions about the data.
6. These assumptions are called inductive biases.

> **A powerful neural network does not merely learn from data. Its architecture tells it what kinds of patterns are worth looking for.**

---

# 🧪 Hands-on challenge

Build a tiny image classifier.

Train it on objects placed in the center.

Then test it with the same objects shifted left and right.

Measure accuracy.

Now compare that with a model that uses convolution.

The experiment will turn the abstract word **invariance** into something you can see.