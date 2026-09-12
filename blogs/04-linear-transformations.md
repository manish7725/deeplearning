# Chapter 04 — A Matrix Can Transform Space

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Laboratory

**[📓 Notebook](https://github.com/manish7725/deeplearning/blob/reorg/class8-to-phd-curriculum/notebooks/04-linear-transformations.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/reorg/class8-to-phd-curriculum/notebooks/04-linear-transformations.ipynb)**

The notebook is the laboratory: calculate first, visualize second, change one variable, and explain what happened.

## 🧭 Where we are

**Came from:** Chapter 03 taught us that matrix multiplication is a grid of dot products.

**Today:** we discover that the same multiplication can be understood geometrically: a matrix transforms vectors and therefore transforms space.

**Next:** Chapter 05 turns this transformation into the smallest trainable neural network.

---

## 1. A matrix is an instruction sheet

Start with

$$
\mathbf{x}=\begin{bmatrix}1\\2\end{bmatrix},\qquad
A=\begin{bmatrix}2&0\\0&2\end{bmatrix}.
$$

Multiply:

$$
A\mathbf{x}=\begin{bmatrix}2\\4\end{bmatrix}.
$$

The vector became twice as long. The matrix performed a **scale** operation.

Instead of thinking of $A$ as four mysterious numbers, read it as two instructions:

$$x'_1=2x_1,\qquad x'_2=2x_2.$$

That is the beginning of geometric linear algebra.

---

## 2. The columns tell the story

Let

$$
A=\begin{bmatrix}2&1\\0&2\end{bmatrix},\qquad
\mathbf{x}=\begin{bmatrix}3\\4\end{bmatrix}.
$$

Then

$$
A\mathbf{x}
=3\begin{bmatrix}2\\0\end{bmatrix}
+4\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}10\\8\end{bmatrix}.
$$

The output is a combination of the columns of $A$.

This gives us an important mental model:

> **A matrix transforms a vector by mixing and scaling the matrix's columns.**

Later, this becomes the language of learned representations and attention projections.

---

## 3. Scaling, reflecting and swapping

### Scaling

$$
\begin{bmatrix}3&0\\0&2\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
=
\begin{bmatrix}3\\2\end{bmatrix}.
$$

Horizontal distances are multiplied by 3; vertical distances by 2.

### Reflection across $y=x$

$$
\begin{bmatrix}0&1\\1&0\end{bmatrix}
\begin{bmatrix}2\\5\end{bmatrix}
=
\begin{bmatrix}5\\2\end{bmatrix}.
$$

### Rotation

A rotation by angle $\theta$ is

$$
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

For $90^\circ$:

$$
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\qquad
R\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0\\1\end{bmatrix}.
$$

The arrow pointing right now points up.

---

## 4. What does *linear* actually mean?

A transformation $T$ is linear when it preserves addition and scaling:

$$
T(\mathbf{x}+\mathbf{y})=T(\mathbf{x})+T(\mathbf{y})
$$

and

$$
T(c\mathbf{x})=cT(\mathbf{x}).
$$

Matrix multiplication satisfies both rules.

There is also a useful combined form:

$$
T(a\mathbf{x}+b\mathbf{y})=aT(\mathbf{x})+bT(\mathbf{y}).
$$

So a linear transformation preserves **linear combinations**.

---

## 5. A subtle but important correction: $Wx+b$

Neural networks usually calculate

$$
\mathbf{z}=W\mathbf{x}+\mathbf{b}.
$$

$W\mathbf{x}$ is linear. Adding a fixed bias shifts the result, so the complete operation is generally called **affine**, not linear.

Then an activation function gives

$$
\mathbf{a}=\sigma(\mathbf{z}).
$$

This distinction matters because precision now prevents confusion later.

---

## 6. Why nonlinear activation is unavoidable

Suppose we stack two linear transformations:

$$
\mathbf{y}=A_2(A_1\mathbf{x}).
$$

Associativity gives

$$
\mathbf{y}=(A_2A_1)\mathbf{x}.
$$

So two linear layers collapse into one linear transformation.

Even affine layers can be combined into one larger affine transformation.

Therefore, depth alone is not enough. A neural network needs a nonlinear operation between transformations if it is to represent genuinely nonlinear relationships.

That is the bridge to Chapter 05.

---

## 7. Transform a whole picture, not just one arrow

Imagine a square made from many points. Apply the same matrix to every point:

```text
original square → matrix A → transformed shape
```

A scaling matrix stretches it. A rotation turns it. A shear slants it.

This is why the best notebook experiment is not one vector. It is a **grid of points**.

### Interactive playground specification

The repository's interactive playground should eventually let the student drag sliders for $a,b,c,d$ in

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
$$

while watching the coordinate grid transform in real time.

Useful controls:

- matrix entries $a,b,c,d$
- reset to identity
- rotation angle
- show basis vectors
- show determinant
- show before/after lengths and angles

The static equation must remain readable even when JavaScript is unavailable.

---

## 8. Determinant: does the transformation squash space?

For

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},
$$

the determinant is

$$
\det(A)=ad-bc.
$$

For a 2D transformation, $|\det(A)|$ tells us how areas scale.

Example:

$$
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
\Rightarrow \det(A)=6.
$$

A unit square becomes a rectangle with area 6.

If

$$
\det(A)=0,
$$

space has been collapsed into a lower-dimensional shape. The transformation is not invertible.

We will return to this idea when studying rank and information loss.

---

## 9. Composition: many small transformations

Suppose $A$ rotates a point and $B$ scales it.

Doing $A$ first and then $B$ gives

$$
B(A\mathbf{x})=(BA)\mathbf{x}.
$$

Notice the order:

$$
BA\neq AB
$$

in general.

This is one of the first places where matrix multiplication stops looking like ordinary multiplication.

> **The rightmost transformation happens first.**

That single sentence will save you from many transformer and neural-network shape mistakes later.

---

## 10. Hand calculation challenge

Calculate

$$
A\mathbf{x}
$$

for

$$
A=\begin{bmatrix}1&2\\3&0\end{bmatrix},
\qquad
\mathbf{x}=\begin{bmatrix}4\\5\end{bmatrix}.
$$

Step by step:

$$
\begin{aligned}
x'_1&=1(4)+2(5)=14\\
x'_2&=3(4)+0(5)=12.
\end{aligned}
$$

Therefore

$$
A\mathbf{x}=\begin{bmatrix}14\\12\end{bmatrix}.
$$

Before using Python, predict this result yourself.

---

## 11. Scientist's experiment

Use a grid of points and compare these matrices:

$$
I=\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
S=\begin{bmatrix}2&0\\0&1\end{bmatrix},
\quad
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\quad
H=\begin{bmatrix}1&1\\0&1\end{bmatrix}.
$$

For each one, record:

1. What happened to the basis vectors?
2. What happened to area?
3. Did angles stay the same?
4. Is the transformation invertible?
5. Can you identify the geometric operation before looking at the plot?

This is how we turn algebra into an experiment.

---

## 12. Failure modes and misconceptions

### ❌ “Every matrix multiplication is a linear transformation.”
Only multiplication by a matrix defines a linear map. Adding a bias produces an affine map.

### ❌ “The order of matrices does not matter.”
Usually false: $AB\neq BA$.

### ❌ “A matrix always preserves distances.”
Only special matrices, such as orthogonal rotation/reflection matrices, preserve Euclidean lengths.

### ❌ “A zero determinant means every output is zero.”
No. It means the transformation loses at least one dimension of information.

### ❌ “A neural network becomes powerful just by adding more linear layers.”
Without nonlinearities, stacked linear transformations collapse into one linear transformation.

---

## 13. Exercises

### Level A — intuition

1. Explain matrix multiplication as a transformation in your own words.
2. Predict what $\begin{bmatrix}2&0\\0&1\end{bmatrix}$ does to a square.
3. Draw the effect of a reflection across $y=x$.

### Level B — calculation

4. Calculate a $2\times2$ matrix times a vector by hand.
5. Calculate the determinant of three matrices.
6. Verify $BA\mathbf{x}$ by first calculating $A\mathbf{x}$ and then $B(A\mathbf{x})$.

### Level C — coding

7. Implement a transformation with NumPy.
8. Plot a grid before and after transformation.
9. Add sliders for matrix entries.

### Level D — deep learning

10. Explain why $W\mathbf{x}+\mathbf{b}$ is affine.
11. Prove that two linear layers without activation collapse into one linear layer.
12. Explain why nonlinear activation is needed.

---

## 🎯 Mastery gate

You are ready for Chapter 05 when you can:

- multiply a matrix by a vector without a library;
- explain the geometric meaning of the columns;
- distinguish linear from affine transformations;
- explain why matrix order matters;
- interpret determinant as area scaling in 2D;
- visualize a transformation of an entire grid;
- explain why neural networks need nonlinear activation.

## 🔬 Research bridge

At graduate level, this simple idea expands into:

- linear operators on high-dimensional spaces;
- eigenvectors and invariant subspaces;
- singular values and conditioning;
- representation transformations;
- equivariance and symmetry;
- learned feature spaces.

A research question to keep:

> **Which transformations preserve the information that a learning system actually needs?**

That question eventually connects linear algebra to representation learning, robustness and architecture design.

## What to remember

$$
\boxed{\mathbf{x}\xrightarrow{W}W\mathbf{x}\xrightarrow{+\mathbf{b}}W\mathbf{x}+\mathbf{b}\xrightarrow{\sigma}\sigma(W\mathbf{x}+\mathbf{b})}
$$

A neural-network layer is not magic. It is a sequence of numerical transformations.

Next, we build the smallest possible trainable version of this machine.

> **Chapter 05 — Meet the Smallest Neural Network.**
