# Lecture 03 — Matrices: The Spreadsheet of Mathematics

> **The Big Question:** How can we hold many vectors as one object and compute with all of them at once?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2003%20-%20Matrices%3A%20The%20Spreadsheet%20of%20Mathematics/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

A vector lets us describe one house with many numbers.

Great. But what if we have **10,000 houses**, each with 20 features?

Handling every vector one at a time would be painful.

> **Can we put all those numbers together and work with them at once?**

**Next → Matrices.**

## 1. The Problem: One House at a Time Does Not Scale

Our four houses are still:

| House | Rooms | Area (sq ft) | Price (₹ lakh) |
|---|---:|---:|---:|
| A | 2 | 800 | 9 |
| B | 2 | 1200 | 11 |
| C | 3 | 900 | 11.5 |
| D | 4 | 1600 | 17 |

Chapter 2 gave us

$$
\hat y=\mathbf w\cdot\mathbf x+b.
$$

For four houses we can write four equations. For four million houses, we cannot. The formula itself would grow with the dataset.

> 🧠 **Think** — Chapter 2 had exactly the same problem one level lower. We could not write one term for every feature, so we introduced a vector. Now we cannot write one equation for every example. We need the same compression idea again.

---

## 2. What Would a Solution Need?

A useful object must:

1. hold many feature vectors;
2. keep each example separate;
3. produce all predictions in one mathematical expression;
4. reduce to Chapter 2 when there is only one example;
5. expose shape mistakes before they become silent bugs.

The fifth requirement is important. Large numerical programs fail surprisingly often because two axes were confused.

---

## 3. First Attempt: A Python List of Vectors

We could store

$$
[\mathbf x_A,\mathbf x_B,\mathbf x_C,\mathbf x_D].
$$

That is perfectly good **storage**. It is not yet the mathematical object we need.

We still have to loop:

```text
for each house:
    compute w · x + b
```

The loop works, but the mathematics has no compact name for it.

> ⚠️ **A tempting wrong idea**
>
> *"A matrix is just a 2-D list."*
>
> A 2-D list stores numbers. A matrix also comes with operations—multiplication, transpose, determinant, inverse when it exists, and a shape discipline. The algebra is the important part.

---

## 4. The Discovery: Stack the Vectors

Put the four house vectors into rows:

$$
X=
\begin{bmatrix}
2&800\\
2&1200\\
3&900\\
4&1600
\end{bmatrix}.
$$

Now one symbol holds the dataset.

| Level | The same idea |
|---|---|
| 💡 **Intuition** | A spreadsheet: one row per house, one column per feature. |
| ✏️ **Numbers** | Row 3 is $[3,900]$: house C. Column 2 is every house's area. |
| 🎓 **Abstraction** | $X\in\mathbb R^{n\times d}$: $n$ examples, $d$ features. |

The entry $X_{ij}$ means **row $i$, column $j$**.

For example,

$$
X_{32}=900.
$$

Learn that order now. It will appear everywhere later.

> 📜 **History Lens — The matrix becomes an object**
>
> Rectangular numerical tables are ancient; Chinese mathematicians used them for systems of equations centuries before modern linear algebra. In 1850, James Joseph Sylvester introduced the word *matrix*. In 1858, Arthur Cayley developed an algebra of matrices in which arrays could be multiplied as objects. That shift—from a table we write in to an object we calculate with—is exactly the shift we need here.

---

## 5. Matrix × Vector: Many Dot Products at Once

We want each row of $X$ to meet the same weight vector.

So define:

$$
X\mathbf w=
\begin{bmatrix}
\text{row}_1\cdot\mathbf w\\
\text{row}_2\cdot\mathbf w\\
\text{row}_3\cdot\mathbf w\\
\text{row}_4\cdot\mathbf w
\end{bmatrix}.
$$

Take

$$
\mathbf w=\begin{bmatrix}2\\0.005\end{bmatrix},\qquad b=1.
$$

Then

$$
X\mathbf w=
\begin{bmatrix}
8\\10\\10.5\\16
\end{bmatrix}
$$

and therefore

$$
\boxed{\hat{\mathbf y}=X\mathbf w+b}
=
\begin{bmatrix}9\\11\\11.5\\17\end{bmatrix}.
$$

One line has replaced four separate calculations.

The operation did not create a new idea. It **named a repeated dot product**.

---

## 6. The Shape Rule

The data matrix has shape $(n,d)$. The weight vector has shape $(d,)$.

$$
X\in\mathbb R^{n\times d},\qquad \mathbf w\in\mathbb R^d
$$

so

$$
X\mathbf w\in\mathbb R^n.
$$

Think visually:

```text
(n × d) · (d,) → (n,)
  ↑       ↑
  └── must match ──┘
```

The shared feature dimension is the axis that gets summed away. The number of examples survives.

This becomes our first **shape debugging rule**:

> **Inner dimensions must agree. The outer dimensions survive.**

---

## 7. The Problem Grows: Many Outputs

Suppose the office wants three outputs for every house:

1. market value;
2. insurance value;
3. tax assessment.

Each output has its own weight vector.

Stack those vectors as **columns**:

$$
W=
\begin{bmatrix}
2&0&1.2\\
0.005&0.004&0.003
\end{bmatrix}.
$$

Its shape is $2\times3$.

- 2 rows = two input features;
- 3 columns = three questions we want answered.

Now the natural operation is matrix × matrix.

---

## 8. Matrix × Matrix Is "Every Question for Every Example"

Define the entry in row $i$, column $j$ by

$$
(XW)_{ij}=\sum_{k=1}^{d}X_{ik}W_{kj}.
$$

Read that sentence in plain English:

> Take one house row, take one output column, multiply matching feature values, and add them.

For house A's tax value:

$$
(XW)_{13}=2(1.2)+800(0.003)=4.8.
$$

Add the tax bias $0.6$:

$$
4.8+0.6=5.4.
$$

All outputs at once are

$$
\hat Y=XW+\mathbf b
$$

with shape

$$
(n\times d)(d\times m)=(n\times m).
$$

For our four houses:

$$
(4\times2)(2\times3)=(4\times3).
$$

That shape is already a story: **four examples, three answers per example**.

---

## 9. Why the Strange Multiplication Rule?

A common question is:

> Why multiply a row by a column and then add?

Because the problem itself asks for a **weighted sum**.

The dot product from Chapter 2 is exactly

$$
\text{weighted contributions} \rightarrow \text{one result}.
$$

Matrix multiplication simply repeats that pattern for every row and every output.

Element-wise multiplication is different:

$$
A\odot B
$$

means matching entries multiply but do **not** sum. That can be useful, but it solves a different problem.

> ⚠️ **A tempting wrong idea**
>
> *"Matrix multiplication should mean multiplying matching cells."*
>
> That is element-wise multiplication. The matrix product exists because composition of dot products needs the sum.

---

## 10. Transpose: Turn Rows Into Columns

Sometimes our information is arranged in the wrong direction.

The **transpose** swaps row and column indices:

$$
X^T_{ij}=X_{ji}.
$$

For

$$
X=
\begin{bmatrix}
2&800\\
2&1200\\
3&900\\
4&1600
\end{bmatrix}
$$

we have

$$
X^T=
\begin{bmatrix}
2&2&3&4\\
800&1200&900&1600
\end{bmatrix}.
$$

Shape changes from $4\times2$ to $2\times4$.

This is not cosmetic. In training we will repeatedly use expressions such as

$$
X^T\mathbf e
$$

because the transpose points the error back into parameter space.

---

## 11. Matrix Multiplication Is Not Commutative

Ordinary numbers satisfy

$$
3\times5=5\times3.
$$

Matrices generally do not.

Take

$$
A=\begin{bmatrix}1&1\\0&1\end{bmatrix},
\qquad
B=\begin{bmatrix}1&0\\1&1\end{bmatrix}.
$$

Then

$$
AB=\begin{bmatrix}2&1\\1&1\end{bmatrix},
\qquad
BA=\begin{bmatrix}1&1\\1&2\end{bmatrix}.
$$

So

$$
\boxed{AB\ne BA}.
$$

The reason will become clearer in Chapter 4: **the matrices represent transformations, and changing the order changes which transformation happens first.**

---

## 12. The Shape Ladder

Deep learning moves constantly between different numbers of dimensions:

| Object | Example | Shape |
|---|---|---|
| Scalar | one price | `()` |
| Vector | one house | `(2,)` |
| Matrix | many houses | `(4,2)` |
| 3-D tensor | many images | `(batch,height,width)` |
| 4-D tensor | colour image batch | `(batch,channels,height,width)` |

A matrix is a rank-2 tensor. A vector is a rank-1 tensor.

Later, the same ideas extend naturally to high-dimensional tensors.

---

## 13. A Matrix Is Also a Compact Program

The expression

$$
XW
$$

does not merely store values. It **describes a computation**.

It says:

```text
for every example
    for every output
        multiply matching features
        add them
```

A numerical library such as NumPy performs that computation efficiently using optimized kernels.

So vectorization is not magic. It is the same loops expressed as algebra.

---

## 14. Scientist's Experiment: Raw Features vs Scaled Features

Chapter 2 showed why feature scaling matters. Here the matrix form makes the reason easier to see.

Our raw columns have wildly different magnitudes:

- rooms: roughly 2–4;
- area: roughly 800–1600.

Gradient descent sees a loss landscape that is much steeper in the area direction than in the room direction.

Try scaling each feature into roughly $[0,1]$:

$$
\text{rooms}'=\frac{\text{rooms}}4,
\qquad
\text{area}'=\frac{\text{area}}{1600}.
$$

The numerical predictions can remain exactly the same after adjusting the weights, but the **optimization geometry** changes.

> 🧪 In the notebook, train the same model twice: once on raw features and once on scaled features. Compare loss curves and number of steps.

The goal is not to memorize "always normalize." The goal is to understand **why scale changes the shape of the problem**.

---

## 15. Failure Modes

| Failure | Why it happens |
|---|---|
| Wrong output shape | Inner dimensions do not match or axes were confused. |
| `X * W` used instead of `X @ W` | Element-wise multiplication is not a collection of dot products. |
| Rows and columns swapped | The feature axis was mistaken for the example axis. |
| `W @ X` used accidentally | Matrix multiplication is order-sensitive. |
| Huge differences in feature scale | Optimization becomes poorly conditioned. |
| Silent broadcasting | Arrays may be numerically accepted while meaning the wrong thing. |

> ⚠️ **Shape bugs are mathematical bugs.**
>
> Do not fix a shape error by randomly adding `.reshape(...)` until the code runs. First write down what each axis means.

---

## 16. Machine Learning Connection

A dense neural-network layer is built from exactly the operation we developed here:

$$
\boxed{Z=XW+b}
$$

where

- $X$ = a batch of examples;
- $W$ = learned weights;
- $b$ = learned bias;
- $Z$ = a batch of outputs.

The same algebra will later appear in attention, embeddings, convolutions after suitable rearrangement, and many optimization derivations.

The matrix is therefore not a side topic. It is one of the main languages of deep learning.

---

## 17. Exercises

### Level A — intuition

1. Explain why stacking vectors into rows solves the "many houses" problem.
2. In `(4 × 2) @ (2 × 3)`, what do the four and three represent?
3. Why does transpose change shape but not the values themselves?

### Level B — calculation

4. Compute a $2\times3$ matrix times a $3\times1$ vector by hand.
5. Calculate one chosen entry of $XW$ from the house example.
6. Find a pair of matrices for which $AB\ne BA$.

### Level C — coding

7. Implement matrix multiplication with three explicit loops.
8. Compare it with NumPy's `@` operator.
9. Add assertions that check expected shapes before multiplication.

### Level D — deep learning

10. Explain why $X^T e$ can have the shape of a weight vector.
11. Derive the number of parameters in a dense layer with $d$ inputs and $m$ outputs.
12. Explain why a matrix product is better than a Python loop for batch computation.

---

## 18. 🏁 Mastery Gate

You are ready for Chapter 4 when you can:

- represent a dataset as a matrix;
- calculate matrix × vector and matrix × matrix by hand;
- use the inner-dimensions rule;
- explain rows versus columns in a machine-learning dataset;
- use transpose correctly;
- explain why `AB` and `BA` are usually different;
- connect $XW+b$ to a neural-network layer;
- diagnose a shape error from mathematics rather than trial and error.

## 19. What We Discovered

1. A matrix lets us hold many vectors without losing their identity.
2. Matrix × vector is many dot products performed as one operation.
3. Matrix × matrix is every output question applied to every example.
4. The shape rule is not a coding convention; it comes directly from the dot product.
5. Transpose changes the orientation of information and becomes essential in learning algorithms.
6. Matrix multiplication is order-sensitive because it represents structured computation, not ordinary scalar multiplication.
7. Feature scale changes optimization geometry.

The most important equation is:

$$
\boxed{(n\times d)(d\times m)=(n\times m)}.
$$

And the next question is bigger:

> **What does a matrix do to the space containing these vectors?**

**Next: Chapter 04 — A Matrix Can Transform Space.**
