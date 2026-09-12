# Blog 01 — What Does It Mean for a Machine to Learn?

Imagine you show a friend ten pictures of apples and ten pictures of oranges. After seeing enough examples, your friend can look at a new fruit and say, “That looks like an orange.”

Did you give your friend a rule for every possible orange? No.

Your friend discovered a **pattern**.

That simple idea is at the heart of machine learning.

## 1. Learning from examples

Suppose each fruit has three measurements:

- weight
- color score
- roundness

We can write one fruit as a list of numbers:

`x = [150, 8, 9]`

Here `x` is called a **feature vector**. A feature is simply a measurable property.

If we collect many examples, we get a dataset:

`X = [x₁, x₂, x₃, ..., xₙ]`

Each example also has an answer, such as `apple` or `orange`.

The computer's job is to discover a useful rule connecting the measurements to the answer.

## 2. The mathematical view

We can imagine that a machine-learning model is a function:

`f(x) → y`

The input `x` contains information about an object. The output `y` is the prediction.

For example:

`f([150, 8, 9]) → orange`

But how does the machine discover `f`?

It starts with adjustable numbers called **parameters**.

A tiny model might be:

`ŷ = wx + b`

Here:

- `w` controls how strongly the input matters.
- `b` shifts the answer.
- `ŷ` is the prediction.

The remarkable part is that we can change `w` and `b` until predictions become better.

## 3. Learning is controlled improvement

Suppose the correct answer for an example is `10`, but our model predicts `6`.

The model is wrong by:

`10 - 6 = 4`

We need a numerical way to describe how wrong the model is. That number is called a **loss**.

A simple loss could be:

`L = (y - ŷ)²`

For our example:

`L = (10 - 6)² = 16`

A better model should make the loss smaller.

So machine learning can be viewed as a repeated game:

**predict → measure error → change parameters → predict again**

## 4. Why mathematics matters

A machine does not understand the word “better.” It understands numbers.

Mathematics lets us turn ideas into quantities:

- vectors represent information
- functions represent models
- loss measures mistakes
- derivatives tell us how parameters should move
- matrices let us perform many calculations together

This is why deep learning is not magic. Underneath the impressive applications are mathematics, data, and computation.

## 5. The big picture

Our journey will follow this path:

`Data → Model → Prediction → Loss → Mathematics → Update → Better Model`

We will eventually build neural networks from simple multiplication and addition.

And that is the first important idea:

> **A learning machine is a mathematical system whose parameters improve by using examples and measuring mistakes.**

### Think like a scientist

If you wanted a computer to recognize dogs and cats, what measurements would you give it? Size? Ear shape? Fur color? Something else?

The quality of those measurements—and the examples you provide—will strongly influence what the machine can learn.