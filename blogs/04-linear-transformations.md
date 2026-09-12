# Blog 04 — A Matrix Can Transform Space

A matrix is not just a rectangle of numbers. It can act like a mathematical machine.

Give it a vector, and it produces a new vector.

That process is called a **linear transformation**.

## 1. Start with a simple vector

Take:

`x = [1, 2]`

Imagine this as an arrow starting at the origin and ending at `(1,2)`.

Now use the matrix:

```text
A = [
    [2, 0],
    [0, 2]
]
```

Multiply:

`Ax = [2,4]`

The arrow has become twice as long.

The matrix performed a **scaling transformation**.

## 2. Another transformation

Consider:

```text
A = [
    [0, 1],
    [1, 0]
]
```

For:

`x = [2, 5]`

we get:

`Ax = [5,2]`

The coordinates swapped.

So the matrix changed the position of the point.

## 3. Why call it linear?

A transformation `T` is linear when it respects two important rules:

`T(x + y) = T(x) + T(y)`

and

`T(cx) = cT(x)`

In simple words:

- adding before transforming gives the same result as transforming first and then adding
- scaling before transforming gives the same result as transforming first and then scaling

These rules allow complicated transformations to be studied using algebra.

## 4. Why should a young programmer care?

Because neural networks repeatedly transform data.

A layer often looks like:

`z = Wx + b`

First the matrix `W` transforms the input `x`.
Then `b` shifts the result.
Then an activation function may change it again.

So a neural network layer is not mysterious. It is a sequence of mathematical transformations.

## 5. A geometric picture in your mind

Imagine a rubber sheet covered with a square grid.

A matrix can stretch the grid, shrink it, rotate it, or shear it.

Every point on the sheet moves according to the same mathematical rule.

That is why linear algebra is so important in machine learning: it gives us a language for transforming information.

## 6. The deeper connection

A feature vector is a representation of an object.

A matrix can transform that representation.

A neural network can apply many such transformations one after another.

This gives us the next major idea:

> **A neural network can be viewed as a machine that repeatedly transforms representations until useful patterns appear.**