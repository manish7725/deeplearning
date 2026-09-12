# Blog 22 — Can a Neural Network Learn Any Shape?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine drawing a road on a giant sheet of paper.

A ruler can draw a straight road.

But what if the road bends?

You could use many short straight pieces:

```text
────╮
    ╰────╮
         ╰────
```

A neural network does something surprisingly similar.

It combines simple mathematical pieces to build complicated functions.

This idea is called **function approximation**.

---

# 1. What is a function?

A function is a machine that turns an input into an output.

$$
f(x)=y
$$

For example:

$$
f(x)=2x+1
$$

If $x=3$:

$$
f(3)=7
$$

A neural network is also a function:

$$
\hat y=f_\theta(x)
$$

The interesting part is that the network can **learn the function from examples**.

---

# 2. One straight line is limited

Suppose we use:

$$
y=wx+b
$$

It can draw a straight line.

But suppose the target looks like a hill:

```text
       ●
     ●   ●
   ●       ●
 ●           ●
```

One straight line cannot describe it well.

So we need something more flexible.

---

# 3. Add a bend

A neuron can calculate:

$$
z=wx+b$$

and then apply an activation function:

$$
h=\sigma(z)
$$

The activation creates non-linearity.

Now we can combine many neurons:

```text
x
│
├── neuron 1 ──┐
├── neuron 2 ──┤
├── neuron 3 ──┼── next layer
└── neuron 4 ──┘
```

Each neuron gives us another small building block.

Many building blocks can make a much more complicated shape.

---

# 4. Width and depth

There are two simple ways to make a network bigger.

### Width

More neurons in one layer:

```text
● ● ● ● ● ●
```

### Depth

More layers:

```text
● ●
 ↓
● ● ●
 ↓
● ● ●
 ↓
●
```

Width gives us more building blocks at once.

Depth lets us **compose** transformations.

Composition means:

$$
f(g(x))
$$

One function acts on the output of another.

---

# 5. The surprising theorem

There is a famous mathematical result called the **universal approximation theorem**.

Very roughly, it says that under suitable conditions, a neural network with enough hidden units and a suitable non-linear activation can approximate a very broad class of continuous functions on a bounded region as closely as we want.

That does **not** mean:

> “A small neural network can solve every problem.”

It means something more subtle:

> **Neural networks are flexible enough to represent very complicated functions.**

This is one reason they are useful.

---

# 6. Why depth matters

Imagine building a complicated pattern with Lego blocks.

You could place thousands of blocks in one huge layer.

Or you could build small structures and place them on top of each other.

Deep networks use the second idea:

```text
simple transformation
        ↓
another transformation
        ↓
another transformation
        ↓
complex transformation
```

MIT's deep-learning course studies this question formally through approximation theory, including universal approximation and whether increasing depth improves expressive power. citeturn0search7

For a Class 7 learner, the key idea is enough:

> **Depth lets simple transformations become a complicated transformation.**

---

# 7. A picture example

Suppose we want to describe a complicated curve.

```text
input x
   ↓
small feature 1
small feature 2
small feature 3
   ↓
combine features
   ↓
new features
   ↓
combine again
   ↓
final curve
```

This is why a neural network is often described as a **function builder**.

---

# 🧪 Think Like a Scientist

Try fitting these two datasets:

### Dataset A

$$
y=2x+1
$$

A single linear neuron should work.

### Dataset B

$$
y=x^2
$$

A straight line cannot represent the curve exactly over a large range.

Ask:

> What does adding a non-linear activation allow the network to do?

That question leads directly to the expressive power of neural networks.

---

# 🧠 What you should remember

1. A neural network is a mathematical function.
2. A single linear layer is limited.
3. Activations introduce non-linearity.
4. Many neurons provide many building blocks.
5. Multiple layers compose simple transformations.
6. Approximation theory studies how well networks can represent functions.
7. The universal approximation theorem explains why neural networks can be extremely expressive.

> **A deep network is not just a bigger calculator. It is a flexible function builder.**

---

# 🧪 Hands-on challenge

Create three models:

```text
Model A → one linear layer
Model B → one hidden layer + ReLU
Model C → three hidden layers + ReLU
```

Try fitting a curve such as:

$$
y=x^2
$$

Plot predictions from all three.

Do not worry if the result is imperfect.

Your job is to observe how **architecture changes what the model can represent**.