# Blog 35 — Why Does the Shape of a Neural Network Matter?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Suppose two students receive the same puzzle.

One student knows that the puzzle is about numbers.

The other knows it is about a map.

They will look for different patterns.

Neural-network architectures work similarly.

The structure of a model influences the kinds of patterns it can learn easily.

This is called **architectural bias** or **inductive bias**.

---

# 1. Same data, different assumptions

Consider an image.

Nearby pixels are often related.

A CNN uses this idea.

Consider a graph.

Relationships between nodes are important.

A GNN uses this idea.

Consider a sequence.

Order matters.

An RNN or Transformer can use this idea.

So:

```text
image      → local structure
sequence   → order / interaction
graph      → relationships
```

---

# 2. Bias is not always bad

The word “bias” can sound negative.

Here it means an assumption that helps learning.

Suppose you know that nearby houses on a street often have similar weather.

That assumption can help you make a prediction.

But if you use it where it is false, it can hurt.

Architectural bias works the same way.

---

# 3. Why shared weights matter

CNNs reuse the same filter across an image.

That means the network assumes:

> “A useful local pattern may be useful in many places.”

This reduces the number of parameters compared with giving every position a completely separate detector.

It also gives convolution useful translation structure.

---

# 4. Why attention is different

A Transformer can compare distant tokens.

In:

```text
The dog chased the ball because it was excited.
```

the word “it” may need information from far away.

Attention lets positions interact directly.

So the architecture contains a different assumption:

> “Important relationships may occur between distant elements.”

---

# 5. Representation is shaped by architecture

Suppose two models see the same image.

A CNN may naturally organize information around local patterns.

A Vision Transformer may organize information through attention between image patches.

Both can learn powerful representations.

But the paths by which they learn are different.

MIT's course explicitly studies how architecture influences learned representations as a separate theory topic. citeturn1search0

---

# 🧠 The big idea

Architecture answers a hidden question:

> **What kinds of relationships should this model find easy to learn?**

That question is much deeper than:

> “Which layer should I use?”

---

# 🧪 Think Like a Scientist

Train two models on the same task.

Change only the architecture.

Compare:

```text
training speed
validation accuracy
number of parameters
robustness
```

You are testing how architectural assumptions influence learning.

---

# 🧠 What you should remember

1. Architecture creates inductive bias.
2. Inductive bias can make some patterns easier to learn.
3. CNNs emphasize local structure and shared filters.
4. GNNs emphasize relationships.
5. Transformers allow flexible interactions between elements.
6. Architecture influences the representations a model learns.

> **Choosing an architecture is like choosing a pair of glasses: it changes which patterns are easier to see.**