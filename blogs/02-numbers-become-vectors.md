# Blog 02 — How Numbers Become Vectors

A computer does not see a red apple the way you do. It sees numbers.

That sounds boring, but it is actually powerful: once something becomes numbers, mathematics can work with it.

## 1. One object, many measurements

Suppose we describe a fruit using:

`weight = 150`

`roundness = 9`

`color = 8`

We can collect them into one object:

`x = [150, 9, 8]`

This is a **vector**.

You can think of a vector as a box containing several related numbers.

## 2. Why put numbers together?

Imagine three students:

`A = [8, 7]`

`B = [9, 8]`

`C = [3, 4]`

Suppose the first number means mathematics marks and the second means science marks.

A and B are similar. C is quite different.

We can measure similarity using distance.

For two-dimensional vectors:

`distance = √((x₁-y₁)² + (x₂-y₂)²)`

For A and B:

`distance = √((8-9)² + (7-8)²)`

`= √(1 + 1)`

`= √2`

A small distance means the students are close in this mathematical space.

## 3. A vector is also a point

The vector `[8, 7]` can be drawn as a point on a graph.

The first number tells us how far to move horizontally. The second tells us how far to move vertically.

So a vector can be both:

- a list of numbers
- a point in a mathematical space

This idea becomes extremely important in machine learning.

## 4. What if we have 100 features?

A fruit might have 100 measurements. A photograph can have millions of pixel values. A language model can work with very large numerical representations.

We simply extend the vector:

`x = [x₁, x₂, x₃, ..., x₁₀₀]`

We may not be able to draw a 100-dimensional space, but the mathematics still works.

Dimensions are not limited by what our eyes can visualize.

## 5. Vectors can be transformed

Suppose:

`x = [2, 3]`

and we multiply every component by 2:

`2x = [4, 6]`

The vector became twice as long.

Now suppose we add another vector:

`a = [1, 5]`

Then:

`x + a = [2+1, 3+5] = [3, 8]`

These simple operations are the building blocks of much more complicated systems.

## 6. The programmer's view

In Python, a vector can be represented using NumPy:

```python
import numpy as np

x = np.array([150, 9, 8])
print(x)
```

The computer now has a mathematical object that we can multiply, add, compare, and transform.

## 7. The deeper idea

When we convert something into numbers, we are creating a **representation**.

A good representation makes the important patterns easier to discover.

That gives us a powerful principle:

> **Machine learning often begins by finding a useful numerical representation of the real world.**

Next, we will see what happens when many vectors are placed together.