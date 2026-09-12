# Blog 11 — Tensors: Numbers in Many Dimensions

You have already met vectors and matrices.

A vector is like a list.
A matrix is like a table.

A **tensor** is a general structure that can organize numbers in many dimensions.

## 1. One dimension

```text
[1, 2, 3]
```

This is a vector.

Its shape is:

`(3)`

## 2. Two dimensions

```text
[
 [1, 2, 3],
 [4, 5, 6]
]
```

This is a matrix.

Its shape is:

`(2, 3)`

Two rows, three columns.

## 3. Three dimensions

Now imagine several matrices stacked together:

```text
[
  [[1,2], [3,4]],
  [[5,6], [7,8]]
]
```

Its shape is:

`(2, 2, 2)`

You can think of this as two tables, each containing two rows and two columns.

## 4. Images are naturally tensors

A grayscale image can be represented as:

`height × width`

A color image often has:

`height × width × channels`

For example:

`224 × 224 × 3`

The 3 channels can represent red, green, and blue.

A batch of 32 images becomes:

`32 × 224 × 224 × 3`

The first number tells us how many images we have.

## 5. Why deep learning loves tensors

Neural networks process huge collections of numbers.

GPUs are especially good at performing the same mathematical operations on many numbers at once.

Tensors provide the structure needed to organize those numbers.

## 6. PyTorch

```python
import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

print(x.shape)
print(x.ndim)
```

You will see:

`torch.Size([2, 3])`

and:

`2`

So the tensor has two dimensions and shape `(2,3)`.

## 7. The key lesson

Do not think of a tensor as something mysterious.

It is simply a disciplined way to organize numbers.

> **Vectors, matrices, and higher-dimensional tensors are different shapes of the same basic idea: structured numerical information.**