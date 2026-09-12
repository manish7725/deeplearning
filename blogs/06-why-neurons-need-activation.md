# Blog 06 — Why Does a Neuron Need an Activation Function?

Our neuron so far is:

`z = wx + b`

This is useful, but it has a surprising limitation.

A purely linear system cannot learn every kind of pattern.

## 1. A simple line

If:

`y = 2x + 1`

then the graph is a straight line.

No matter how complicated our calculations become, if every layer only performs linear transformations, the entire network can still be simplified into one big linear transformation.

That means we need something that bends the rules.

## 2. Enter the activation function

After calculating `z`, we apply a function:

`a = f(z)`

The function `f` is called an **activation function**.

One famous activation is ReLU:

`ReLU(z) = max(0, z)`

So:

- ReLU(-3) = 0
- ReLU(2) = 2
- ReLU(7) = 7

It removes negative values but keeps positive ones.

## 3. Why is bending useful?

Imagine trying to separate two groups of points with a straight line.

Sometimes it works.

But suppose the red points are in the middle of a circle and blue points are outside the circle.

One straight line cannot separate them perfectly.

A neural network needs to create more complicated boundaries.

Activation functions introduce the non-linearity that makes this possible.

## 4. A tiny network

Imagine:

`input → matrix multiplication → ReLU → matrix multiplication → output`

The first layer can transform the input.
The ReLU can bend the representation.
The next layer can combine the transformed features.

Repeating this process lets the network represent increasingly complicated functions.

## 5. Other activations

You will eventually meet:

- sigmoid
- tanh
- ReLU
- GELU
- softmax

Each has a different mathematical shape and purpose.

For now, remember the key idea:

> **Activation functions prevent a neural network from collapsing into one giant straight-line calculation.**

## 6. A programmer's experiment

```python
import torch

x = torch.tensor([-3.0, 2.0, 7.0])
print(torch.relu(x))
```

The result is:

`tensor([0., 2., 7.])`

A tiny mathematical rule has become a reusable program.

And now our neural network has learned its first trick: it can create non-linear representations.