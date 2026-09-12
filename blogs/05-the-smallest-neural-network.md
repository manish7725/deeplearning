# Blog 05 — Meet the Smallest Neural Network

Let us build a neural network using almost nothing: multiplication, addition, and one adjustable number.

## 1. A neuron as a calculator

Suppose the input is:

`x = 3`

The neuron has a weight:

`w = 2`

It calculates:

`z = wx`

So:

`z = 2 × 3 = 6`

That is a neuron doing its first job.

## 2. Add a bias

Now introduce another number called a **bias**:

`z = wx + b`

Suppose:

`w = 2`

`x = 3`

`b = 1`

Then:

`z = 2×3 + 1 = 7`

The weight controls how strongly the input matters. The bias lets the neuron shift its output.

## 3. Two inputs

Real problems usually have many features.

Suppose:

`x = [2, 3]`

and:

`w = [4, 5]`

The neuron computes a dot product:

`z = 2×4 + 3×5 + b`

If `b = 1`:

`z = 8 + 15 + 1 = 24`

That is a complete neuron.

## 4. Why weights are important

Imagine we are predicting whether a student is likely to enjoy a science experiment.

Maybe:

- curiosity matters a lot
- previous science marks matter somewhat
- drawing ability matters less

The model can represent these differences using weights.

A large positive weight means a feature strongly pushes the result upward.
A small weight means it has less influence.
A negative weight can push the result downward.

## 5. From one neuron to a layer

Now imagine 3 neurons receiving the same input.

Each neuron has different weights:

`neuron 1 → pattern A`

`neuron 2 → pattern B`

`neuron 3 → pattern C`

Together they form a layer.

In matrix form:

`z = Wx + b`

This is why the matrix ideas from our earlier blog suddenly become useful.

## 6. The surprising part

A neuron is not intelligent by itself.

It is a tiny mathematical function.

The intelligence comes from having:

- many parameters
- many examples
- a useful loss function
- an algorithm that improves the parameters

The neuron is the brick. A neural network is the building.

> **Deep learning starts with very simple mathematical operations repeated on a very large scale.**