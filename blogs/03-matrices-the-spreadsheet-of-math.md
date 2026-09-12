# Chapter 03 — Matrices: The Spreadsheet of Mathematics

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Laboratory

**[📓 Open the matching notebook on GitHub](https://github.com/manish7725/deeplearning/blob/reorg/class8-to-phd-curriculum/notebooks/03-matrices-the-spreadsheet-of-math.ipynb)** · **[▶ Open in Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/reorg/class8-to-phd-curriculum/notebooks/03-matrices-the-spreadsheet-of-math.ipynb)**

The notebook is the laboratory for this chapter: calculate every example, change the numbers, visualize the operation, and deliberately create shape errors.

## 🧭 Where we are going

**We came from:** Chapter 02 turned one object into a vector.

**Today:** We need a way to hold **many vectors together** and process them efficiently.

**Next:** Chapter 04 will show that a matrix is more than a table: it can **transform space**.

> A vector describes one object. A matrix organizes many objects and can transform them.

---

## 1. Start with a classroom

Imagine three students. We record two scores for each student:

| Student | Math | Science |
|---|---:|---:|
| Alice | 8 | 7 |
| Bob | 9 | 8 |
| Cara | 3 | 4 |

We can write the whole table as

$$
X =
\begin{bmatrix}
8 & 7\\
9 & 8\\
3 & 4
\end{bmatrix}.
$$

This is a **matrix**.

It has:

- 3 rows → 3 students
- 2 columns → 2 features

So its shape is

$$
X\in\mathbb{R}^{3\times2}.
$$

Think of a matrix as a spreadsheet whose cells can participate in mathematics.

---

## 2. A matrix is a family of vectors

Look at the rows:

$$
[8,7],\quad [9,8],\quad [3,4].
$$

These are exactly the vectors from the previous chapter.

So a matrix is not a mysterious new object. It is a convenient way to organize vectors.

This gives us our first important mental model:

```text
vector → one example
matrix → many examples together
```

There is another useful view. The columns are also vectors:

```text
Math     → [8, 9, 3]
Science  → [7, 8, 4]
```

The same matrix can therefore be viewed **row-wise or column-wise**, depending on the question we are asking.

---

## 3. Shape is part of the meaning

Suppose

$$
X\in\mathbb{R}^{64\times128}.
$$

A common interpretation is:

```text
64 examples × 128 features
```

If a neural network changes those 128 features into 256 features, we might use a weight matrix with shape

$$
W\in\mathbb{R}^{128\times256}.
$$

Then

$$
XW\in\mathbb{R}^{64\times256}.
$$

Notice how the shapes tell a story:

```text
64 examples
      ×
128 old features
      ↓
256 new features
```

In deep learning, a shape error is often not just a programming problem. It can reveal that our mathematical idea is wrong.

---

## 4. The most important rule of matrix multiplication

Take

$$
A\in\mathbb{R}^{m\times n},
\qquad
B\in\mathbb{R}^{n\times p}.
$$

Then

$$
AB\in\mathbb{R}^{m\times p}.
$$

The **inside numbers must match**:

$$
(m\times\boxed n)(\boxed n\times p).
$$

Why?

Because each output number is created by taking a dot product. A dot product requires two lists with the same number of entries.

This is not an arbitrary rule invented by mathematicians. It is forced by the calculation we want to perform.

---

## 5. Build one output number by hand

Let

$$
A=
\begin{bmatrix}
2&3\\
4&5
\end{bmatrix},
\qquad
B=
\begin{bmatrix}
10&1\\
20&2
\end{bmatrix}.
$$

To find the top-left number, take the first row of $A$ and first column of $B$:

$$
2(10)+3(20)=20+60=80.
$$

Top-right:

$$
2(1)+3(2)=2+6=8.
$$

Bottom-left:

$$
4(10)+5(20)=40+100=140.
$$

Bottom-right:

$$
4(1)+5(2)=4+10=14.
$$

Therefore

$$
AB=
\begin{bmatrix}
80&8\\
140&14
\end{bmatrix}.
$$

### The key idea

> **Matrix multiplication is a grid of dot products.**

If you understand that sentence, you understand the core computation behind dense neural-network layers.

---

## 6. From one neuron to many neurons

In Chapter 02 we saw a neuron-like weighted sum:

$$
\mathbf{x}=\begin{bmatrix}2\\3\end{bmatrix},
\qquad
\mathbf{w}=\begin{bmatrix}4\\5\end{bmatrix}.
$$

The weighted sum is

$$
\mathbf{w}^T\mathbf{x}=4(2)+5(3)=23.
$$

That is one neuron processing one example.

Now imagine three neurons:

$$
W=
\begin{bmatrix}
4&5\\
1&2\\
7&-1
\end{bmatrix}.
$$

Multiplying

$$
W\mathbf{x}
$$

produces three weighted sums at once:

$$
\begin{bmatrix}
4&5\\
1&2\\
7&-1
\end{bmatrix}
\begin{bmatrix}2\\3\end{bmatrix}
=
\begin{bmatrix}
23\\8\\11
\end{bmatrix}.
$$

One matrix multiplication has performed the work of **three neurons**.

Now add a batch of 64 examples and hundreds of neurons. The same mathematical pattern scales naturally.

---

## 7. Why computers love this operation

Suppose we have:

$$
X\in\mathbb{R}^{64\times128}
$$

and

$$
W\in\mathbb{R}^{128\times256}.
$$

The result has shape

$$
64\times256.
$$

There are 64 examples and 256 output neurons.

Each of those 64 × 256 output numbers is a dot product of length 128.

So the operation contains

$$
64\times256=16,384
$$

dot products, each involving 128 multiply-add operations.

A modern GPU is designed to perform huge numbers of these regular numerical operations in parallel.

That is one major reason matrix multiplication is at the heart of deep learning hardware.

---

## 8. Matrix multiplication is different from cell-by-cell multiplication

There are two different operations that beginners often confuse.

### Matrix multiplication

$$
AB
$$

uses rows and columns and creates dot products.

### Element-wise multiplication

If two matrices have the same shape,

$$
A\odot B
$$

multiplies corresponding cells.

For example,

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}
\odot
\begin{bmatrix}5&6\\7&8\end{bmatrix}
=
\begin{bmatrix}5&12\\21&32\end{bmatrix}.
$$

But

$$
AB
$$

would give a completely different result.

> Never assume that `*` and `@` mean the same thing in NumPy.

---

## 9. Matrix multiplication is usually not commutative

With ordinary numbers,

$$
2\times3=3\times2.
$$

But matrices generally behave differently:

$$
AB\neq BA.
$$

Sometimes $AB$ exists while $BA$ does not even have a valid shape.

This matters because the **order of transformations matters**.

We will see the geometric reason in the next chapter.

---

## 10. Transpose: turn rows into columns

For

$$
A=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix},
$$

its transpose is

$$
A^T=
\begin{bmatrix}
1&4\\
2&5\\
3&6
\end{bmatrix}.
$$

The shape changes from

$$
2\times3
$$

to

$$
3\times2.
$$

The transpose operation is simple, but it becomes extremely important later in linear algebra, least squares, attention, and matrix calculus.

---

## 11. A complete neural-network layer

A dense neural-network layer is often written as

$$
Z=XW+b.
$$

Here:

- $X$ = input examples
- $W$ = learnable weights
- $b$ = learnable bias
- $Z$ = output before activation

Then an activation function may produce

$$
H=\sigma(Z).
$$

So one layer looks like

```text
many input vectors
       ↓
   matrix X
       ↓
   multiply W
       ↓
    add b
       ↓
  activation σ
       ↓
new representations
```

We have now connected school-level spreadsheets to the computational structure of a neural network.

---

## 12. Hand-calculation challenge

Before opening the notebook, calculate:

$$
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
\begin{bmatrix}
5\\
6
\end{bmatrix}.
$$

First predict the shape.

Then calculate both output values.

Then verify with Python.

### Shape puzzle

Can we multiply

$$
(4\times3)(3\times2)?
$$

Yes:

$$
(4\times3)(3\times2)=(4\times2).
$$

Can we multiply

$$
(4\times3)(4\times2)?
$$

No. The inside dimensions are 3 and 4, so the dot products cannot be formed.

---

## 13. Interactive mathematical playground

The repository should eventually make this chapter **feel** rather than merely read.

### Playground: watch one cell being calculated

A useful interactive visualization would let you choose a row of $A$ and a column of $B$ and then animate:

```text
2 × 10  → 20
3 × 20  → 60
20 + 60 → 80
```

Then highlight the resulting cell in $AB$.

### Playground: shape laboratory

Let a student change

$$
(m\times n)(n\times p)
$$

with sliders and immediately see whether multiplication is valid and what the output shape will be.

### Playground: neural-network batch

Show several input vectors entering a weight matrix and animate the resulting dot products into several neurons.

The static equations above remain the source of truth, so the chapter is still useful without JavaScript.

---

## 14. Python: the calculator becomes a laboratory

```python
import numpy as np

A = np.array([
    [2., 3.],
    [4., 5.]
])

B = np.array([
    [10., 1.],
    [20., 2.]
])

print("A shape:", A.shape)
print("B shape:", B.shape)
print("A @ B:\n", A @ B)
print("A * B:\n", A * B)
```

The two results are intentionally different.

Try predicting both before running the cell.

Then implement matrix multiplication using ordinary Python loops. Your result should agree with `A @ B`.

---

## 15. PyTorch: same mathematics, bigger machines

```python
import torch

X = torch.tensor([
    [2., 3.],
    [4., 5.]
])

W = torch.tensor([
    [10.],
    [20.]
])

print(X @ W)
```

The mathematics is unchanged.

PyTorch adds tools that become important later: automatic differentiation, accelerators, tensor operations, and neural-network abstractions.

Our rule remains:

> **Understand the operation first. Use the framework second.**

---

## 16. Common mistakes

### Mistake 1 — Treating a matrix as just a spreadsheet

It is a table, but it can also represent a linear transformation. That is the next chapter.

### Mistake 2 — Forgetting order

$AB$ and $BA$ are generally different.

### Mistake 3 — Ignoring shapes

Always write the shapes beside your calculation when learning.

### Mistake 4 — Confusing `*` with `@`

Element-wise multiplication and matrix multiplication are different operations.

### Mistake 5 — Memorizing the shape rule without understanding it

Remember the reason:

> **A matrix product is made from dot products, and dot products need matching lengths.**

---

## 🧠 Think Like a Scientist

Take

$$
X\in\mathbb{R}^{100\times20},
\qquad
W\in\mathbb{R}^{20\times50}.
$$

Predict:

1. Is $XW$ valid?
2. What is its shape?
3. How many output numbers are produced?
4. How long is each dot product?
5. What would happen if $W$ had shape $21\times50$?

Do not run code until you have written your predictions.

That habit—**predict → calculate → run → explain**—will become the scientific method of this curriculum.

---

## What you should remember

> **A matrix organizes vectors, and matrix multiplication performs many dot products in a structured way.**

You should now be comfortable with:

- rows and columns;
- matrix shape;
- vectors inside matrices;
- matrix-vector multiplication;
- matrix-matrix multiplication;
- transpose;
- element-wise multiplication vs matrix multiplication;
- why inner dimensions must match;
- why matrices are central to neural-network layers.

The next question is much more exciting:

> **What does a matrix actually do to a point in space?**

That is where multiplication stops looking like a spreadsheet calculation and starts looking like geometry.

**Next → Chapter 04: A Matrix Can Transform Space.**

---

# 🎯 Exercises

### Beginner

1. Write a 3 × 2 matrix representing three students and two features.
2. Identify its rows and columns.
3. Transpose it by hand.
4. Calculate a 2 × 2 matrix product.

### Intermediate

5. Implement matrix multiplication with nested loops.
6. Add assertions for all expected shapes.
7. Compare your implementation with NumPy.
8. Demonstrate that $AB\neq BA$ using a numerical example.

### Advanced

9. Explain matrix multiplication entirely in terms of dot products.
10. Explain why a dense neural-network layer can be represented by $XW+b$.
11. Calculate the number of multiply-add operations for a $512\times1024$ input multiplied by a $1024\times4096$ weight matrix.
12. Investigate how changing the batch size affects the shape and computation.

### Mini-project

Build a tiny **matrix calculator** that accepts two matrices, validates their shapes, computes valid products, and explains the shape rule when multiplication is impossible.

### Research bridge

Later in the curriculum, revisit matrix multiplication when learning:

- linear transformations;
- eigenvalues and eigenvectors;
- SVD and low-rank approximation;
- neural-network layers;
- attention: $QK^T$;
- GPU matrix-multiplication kernels;
- matrix calculus.

The same operation will keep appearing, but each time we will understand one more layer of why it matters.
