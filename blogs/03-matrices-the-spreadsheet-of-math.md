# Blog 03 — Matrices: The Spreadsheet of Mathematics

A vector stores one collection of numbers. What if we have thousands of vectors?

We can arrange them into rows and columns.

That structure is called a **matrix**.

## 1. A tiny dataset

Suppose three students have marks in Mathematics and Science:

```text
X = [
    [8, 7],
    [9, 8],
    [3, 4]
]
```

There are 3 rows and 2 columns.

We say the matrix has **shape `(3, 2)`**.

Rows represent students. Columns represent subjects.

## 2. Why matrices are useful

A matrix lets us describe many examples at once.

Instead of writing:

`student1 = [8, 7]`

`student2 = [9, 8]`

`student3 = [3, 4]`

we write one object:

`X = [[8,7],[9,8],[3,4]]`

Now mathematics can process the whole dataset together.

## 3. Matrix multiplication

Here is one of the most important operations in machine learning.

Take:

```text
X = [
    [2, 3],
    [4, 5]
]
```

and:

```text
W = [
    [10],
    [20]
]
```

Then:

`XW = [[2×10 + 3×20], [4×10 + 5×20]]`

So:

`XW = [[80], [140]]`

Notice something beautiful: one multiplication processed two examples at once.

## 4. Why the inside dimensions matter

For matrix multiplication:

`(m × n) × (n × p) = (m × p)`

The two middle numbers must match.

For our example:

`(2 × 2) × (2 × 1) = (2 × 1)`

The two `2`s in the middle match.

This is not an arbitrary rule. It tells us that the numbers being combined line up correctly.

## 5. A neural network connection

Suppose every student has two features and we want one score.

The weight vector could be:

`w = [10, 20]`

For one student:

`[2,3] · [10,20] = 2×10 + 3×20 = 80`

That dot product is already a tiny neuron.

A neural network is built by performing huge numbers of operations like this.

## 6. Python makes this practical

```python
import numpy as np

X = np.array([[2, 3], [4, 5]])
W = np.array([[10], [20]])

print(X @ W)
```

The `@` operator performs matrix multiplication.

## 7. The important mental model

Think of a matrix as a **machine for organizing many vectors**.

A vector can describe one object.
A matrix can describe many objects.
Matrix multiplication can transform many objects together.

> **Deep learning works largely because computers can perform enormous numbers of matrix operations extremely quickly.**

Next we will turn these numbers into geometry and discover linear transformations.