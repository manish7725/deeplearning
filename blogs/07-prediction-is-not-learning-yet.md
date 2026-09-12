# Blog 07 — Making a Prediction Is Not the Same as Learning

We now have a neuron that can calculate:

`ŷ = f(wx + b)`

It can make a prediction.

But it still does not know how to choose good values for `w` and `b`.

That is where learning begins.

## 1. The guessing game

Suppose the correct answer is:

`y = 10`

Our model predicts:

`ŷ = 7`

The model needs to know that 7 is worse than 9, and 9 is worse than 10.

We need a number that measures the quality of the prediction.

That number is the **loss**.

## 2. Squared error

A simple loss is:

`L = (y - ŷ)²`

For `y = 10` and `ŷ = 7`:

`L = (10 - 7)² = 9`

If the model predicts 9:

`L = (10 - 9)² = 1`

Smaller is better.

## 3. Why square the error?

Suppose the error is `-3`.

If we simply use the error, negative numbers could cancel positive numbers.

Squaring gives:

`(-3)² = 9`

and:

`3² = 9`

Both mistakes become positive.

The square also makes large mistakes hurt more.

## 4. One example is not enough

Suppose we have 100 training examples.

Each one has a loss:

`L₁, L₂, ..., L₁₀₀`

We can calculate the average:

`L = (L₁ + L₂ + ... + L₁₀₀) / 100`

Now the model has one overall score.

Our goal becomes beautifully simple:

> **Find parameters that make the loss as small as possible.**

## 5. A mountain analogy

Imagine the loss is the height of a mountain.

Our parameters determine where we are standing.

We want to walk downhill until we reach a low point.

But how do we know which direction is downhill?

Calculus gives us the answer.

The derivative tells us how a quantity changes.

For many parameters, we use a collection of derivatives called the **gradient**.

That is the next great idea we will study.

## 6. The full learning loop

Our model now has a complete story:

`input → prediction → loss → gradient → parameter update`

Then we repeat.

This loop is the heartbeat of neural-network training.