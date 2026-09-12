# Blog 10 — Backpropagation: Sending the Mistake Backward

A neural network can have many layers.

If the final answer is wrong, how does each earlier layer know what it should change?

The answer is **backpropagation**.

## 1. A tiny two-step network

Imagine:

`x → w₁ → w₂ → prediction`

Let:

`z = w₁x`

and:

`ŷ = w₂z`

Suppose:

`x = 2`

`w₁ = 3`

`w₂ = 4`

Then:

`z = 3×2 = 6`

`ŷ = 4×6 = 24`

Suppose the correct answer is 20.

The prediction is wrong.

## 2. The question

Which parameter caused the mistake?

Both `w₁` and `w₂` matter.

We want to know how the final loss changes when each parameter changes.

This is where the **chain rule** from calculus becomes powerful.

## 3. The chain rule

If:

`a` affects `b`, and `b` affects `c`,

then the effect of `a` on `c` can be calculated through the chain:

`dc/da = (dc/db)(db/da)`

For our network, the loss depends on `w₂`, which depends on `z`, which depends on `w₁`.

So we can calculate the effect step by step.

## 4. Why backward?

The network calculates its prediction from left to right:

`input → layer 1 → layer 2 → output`

But once the loss is known, gradients are efficiently calculated from right to left:

`loss → output → layer 2 → layer 1`

That is backpropagation.

## 5. A simple mental model

Imagine a team builds a robot together.

The robot fails to pick up a ball.

The final failure is observed at the end, but we need to trace backward:

- Did the hand move incorrectly?
- Was the arm positioned incorrectly?
- Was the shoulder command incorrect?

Backpropagation performs a mathematical version of this responsibility tracing.

## 6. The important insight

Backpropagation does not magically discover the answer.

It uses calculus—especially the chain rule—to calculate how much each parameter contributed to the final loss.

Then gradient descent uses those gradients to update the parameters.

So:

**Backpropagation calculates the directions. Gradient descent takes the steps.**

Together, they make neural-network training possible.