# Blog 15 — How Can a Computer Represent the Meaning of a Word?

A computer cannot directly understand the word “king.”

It needs numbers.

But simply assigning `king = 17` and `queen = 42` does not tell us much about meaning.

We want a representation where relationships can be captured mathematically.

That idea leads to **embeddings**.

## 1. Words become vectors

Imagine a tiny vocabulary where each word gets a vector:

`king = [0.9, 0.8, 0.7]`

`queen = [0.9, 0.7, 0.8]`

`apple = [0.1, 0.2, 0.9]`

These numbers are not human-designed definitions. A model can learn useful representations from data.

## 2. Similar things can become nearby

If two words are used in similar contexts, their vectors may become closer in the learned space.

For example:

`doctor` and `nurse`

may end up closer than:

`doctor` and `mountain`.

The exact geometry depends on the model and training process.

## 3. Distance becomes meaningful

We can compare vectors using measures such as Euclidean distance or cosine similarity.

Cosine similarity asks roughly:

“Are these two vectors pointing in a similar direction?”

That lets us turn some questions about language into questions about geometry.

## 4. Why this is powerful

Once words, sentences, images, or other objects become vectors, the machinery of linear algebra becomes available.

We can:

- compare representations
- transform them
- combine them
- feed them into neural networks

## 5. The surprising connection

Earlier we said a vector is a numerical representation.

Embeddings are a sophisticated version of exactly that idea.

The model learns a numerical space in which useful relationships can emerge.

> **An embedding is a learned coordinate system for representing information.**

This prepares us for attention, where a model learns which parts of a representation deserve more focus.