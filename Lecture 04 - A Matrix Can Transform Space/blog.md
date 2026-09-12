# Chapter 04 — A Matrix Can Transform Space

> **The Big Question:** If a matrix multiplies a vector, what is it actually *doing* to that vector?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2004%20-%20A%20Matrix%20Can%20Transform%20Space/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## 🧭 Where We Are

**Previously:** Chapter 3 taught us that matrix multiplication is a grid of dot products.

**Today:** We stop treating a matrix as a spreadsheet and discover its geometric meaning: a matrix can transform vectors and therefore transform the space around them.

**Next:** We use that transformation as the heart of a trainable neuron.

---

## 1. Start With the Simplest Matrix

Take

$$
A=\begin{bmatrix}2&0\\0&2\end{bmatrix},
\qquad
\mathbf x=\begin{bmatrix}1\\2\end{bmatrix}.
$$

Multiply:

$$
A\mathbf x=
\begin{bmatrix}2\\4\end{bmatrix}.
$$

The arrow got twice as long.

The matrix therefore acted like a **scale operation**.

Instead of memorizing a matrix as four numbers, read its equations:

$$
x'_1=2x_1,
\qquad
x'_2=2x_2.
$$

That is the beginning of geometric linear algebra.

---

## 2. The Columns Tell the Story

Consider

$$
A=\begin{bmatrix}2&1\\0&2\end{bmatrix},
\qquad
\mathbf x=\begin{bmatrix}3\\4\end{bmatrix}.
$$

Matrix multiplication gives

$$
A\mathbf x
=3\begin{bmatrix}2\\0\end{bmatrix}
+4\begin{bmatrix}1\\2\end{bmatrix}
=\begin{bmatrix}10\\8\end{bmatrix}.
$$

This reveals a powerful interpretation:

> **The output is a linear combination of the columns of the matrix.**

The entries of $\mathbf x$ tell us how much of each column to take.

So a matrix is not simply changing one coordinate independently. It can **mix** coordinates.

---

## 3. Basis Vectors Make This Even Clearer

The standard basis vectors in 2-D are

$$
\mathbf e_1=\begin{bmatrix}1\\0\end{bmatrix},
\qquad
\mathbf e_2=\begin{bmatrix}0\\1\end{bmatrix}.
$$

Every vector can be written as

$$
\mathbf x=x_1\mathbf e_1+x_2\mathbf e_2.
$$

For example,

$$
\begin{bmatrix}3\\4\end{bmatrix}
=3\mathbf e_1+4\mathbf e_2.
$$

Because matrix multiplication preserves linear combinations,

$$
A\mathbf x
=x_1A\mathbf e_1+x_2A\mathbf e_2.
$$

But $A\mathbf e_1$ is exactly **column 1 of $A$**, and $A\mathbf e_2$ is exactly **column 2**.

So the columns tell us exactly where the basis directions go.

> 🧠 **Mental model:** To understand a 2-D matrix, first ask: *Where does $[1,0]$ go? Where does $[0,1]$ go?* The whole transformation follows from those two answers.

---

## 4. Scaling, Reflection, and Swapping

### Scaling

$$
\begin{bmatrix}3&0\\0&2\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
=
\begin{bmatrix}3\\2\end{bmatrix}.
$$

Horizontal and vertical directions are stretched by different amounts.

### Reflection across $y=x$

$$
\begin{bmatrix}0&1\\1&0\end{bmatrix}
\begin{bmatrix}2\\5\end{bmatrix}
=
\begin{bmatrix}5\\2\end{bmatrix}.
$$

The coordinates swap.

### Rotation

A counter-clockwise rotation through angle $\theta$ is

$$
R(\theta)=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}.
$$

For $90^\circ$,

$$
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}
$$

and therefore

$$
R\begin{bmatrix}1\\0\end{bmatrix}
=\begin{bmatrix}0\\1\end{bmatrix}.
$$

The right-pointing arrow now points up.

---

## 5. A Shear: When Directions Mix

Try

$$
H=\begin{bmatrix}1&1\\0&1\end{bmatrix}.
$$

For

$$
\mathbf x=\begin{bmatrix}2\\1\end{bmatrix},
$$

we get

$$
H\mathbf x=
\begin{bmatrix}3\\1\end{bmatrix}.
$$

The vertical coordinate stayed the same, while the horizontal coordinate picked up some of the vertical coordinate.

This is called a **shear**.

It is a good reminder that matrices can mix coordinates rather than merely scale them.

---

## 6. What Does “Linear” Mean?

A transformation $T$ is **linear** when it preserves addition and scalar multiplication:

$$
T(\mathbf x+\mathbf y)=T(\mathbf x)+T(\mathbf y)
$$

and

$$
T(c\mathbf x)=cT(\mathbf x).
$$

Together,

$$
T(a\mathbf x+b\mathbf y)=aT(\mathbf x)+bT(\mathbf y).
$$

Matrix multiplication satisfies this automatically.

This explains why the columns of a matrix determine the entire transformation: every input is a linear combination of basis vectors, and the transformation preserves that combination.

---

## 7. A Subtle Correction: Neural Networks Use Affine Maps

A common neural-network expression is

$$
\mathbf z=W\mathbf x+\mathbf b.
$$

The $W\mathbf x$ part is linear.

The addition of a fixed bias shifts the result, so the complete operation is generally called **affine**.

This distinction matters.

### Linear

$$
T(\mathbf 0)=\mathbf 0.
$$

### Affine

$$
T(\mathbf x)=W\mathbf x+\mathbf b
$$

may have

$$
T(\mathbf 0)=\mathbf b\neq\mathbf 0.
$$

> ⚠️ **Common mistake:** saying that $W\mathbf x+\mathbf b$ is linear just because it contains a matrix multiplication. The bias changes the mathematical class.

---

## 8. Transform an Entire Grid

A single vector can hide what a transformation is doing.

A better experiment is to transform **many points at once**.

Imagine a square grid:

```text
original space

+---+---+---+
|   |   |   |
+---+---+---+
|   |   |   |
+---+---+---+
|   |   |   |
+---+---+---+
```

Apply a matrix to every point.

- scaling stretches the grid;
- rotation turns it;
- reflection flips it;
- shear slants it.

This is why the notebook uses a complete point grid rather than one arrow.

### Interactive playground

Use the notebook to vary

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix}
$$

and watch the grid change.

The useful controls are:

- $a,b,c,d$;
- rotation angle;
- reset to identity;
- show basis vectors;
- show determinant;
- compare before and after lengths.

The equations in this chapter remain the source of truth when the interactive plot is unavailable.

---

## 9. Composition: One Transformation After Another

Suppose $A$ transforms a point first and $B$ transforms the result second.

Then

$$
\mathbf y=B(A\mathbf x).
$$

Associativity lets us write

$$
\mathbf y=(BA)\mathbf x.
$$

Notice the order:

> **The rightmost transformation happens first.**

This also explains Chapter 3's surprising result:

$$
AB\neq BA.
$$

If one matrix rotates and another scales, doing rotation-then-scale usually differs from scale-then-rotation.

This is the geometric reason matrix multiplication is not commutative.

---

## 10. Determinant: Does the Transformation Lose Space?

For

$$
A=\begin{bmatrix}a&b\\c&d\end{bmatrix},
$$

the determinant is

$$
\boxed{\det(A)=ad-bc}.
$$

In 2-D, the absolute value $|\det(A)|$ tells us how areas scale.

Example:

$$
A=\begin{bmatrix}2&0\\0&3\end{bmatrix}
$$

has

$$
\det(A)=6.
$$

A unit square therefore becomes a shape with area 6.

Now consider

$$
A=\begin{bmatrix}1&2\\2&4\end{bmatrix}.
$$

Its determinant is

$$
1(4)-2(2)=0.
$$

The columns are dependent: the second column is twice the first.

The 2-D region collapses onto a line. Some information is lost.

> 💡 **Geometric meaning:** determinant zero means the transformation crushed at least one dimension.

This will later connect to rank, invertibility, and information loss.

---

## 11. Invertibility: Can We Undo the Transformation?

If a transformation can be reversed, there exists a matrix $A^{-1}$ such that

$$
A^{-1}A=I.
$$

For a $2\times2$ matrix,

$$
A^{-1}
=\frac{1}{ad-bc}
\begin{bmatrix}
 d&-b\\
-c&a
\end{bmatrix}
$$

provided

$$
\det(A)\neq0.
$$

So determinant zero and invertibility are linked:

$$
\boxed{\det(A)=0\quad\Longrightarrow\quad A\text{ is not invertible}.}
$$

The transformation lost information, so there is no unique way to reconstruct the input.

---

## 12. Two Linear Layers Without Activation Collapse

Now connect geometry to deep learning.

Suppose we stack two linear transformations:

$$
\mathbf y=A_2(A_1\mathbf x).
$$

Associativity gives

$$
\mathbf y=(A_2A_1)\mathbf x.
$$

The two layers are equivalent to one larger matrix.

This means:

> **Adding more linear layers does not automatically make a network more expressive.**

It only changes how the same overall linear transformation is factorized.

Even affine layers can be combined into one affine transformation.

That is why a neural network needs something genuinely nonlinear between layers.

---

## 13. Where the Activation Function Enters

A typical neural-network layer has the pattern

$$
\mathbf z=W\mathbf x+\mathbf b
$$

followed by

$$
\mathbf a=\sigma(\mathbf z).
$$

Here $\sigma$ is a nonlinear activation function.

Now the next layer sees

$$
W_2\sigma(W_1\mathbf x+\mathbf b_1)+\mathbf b_2.
$$

This cannot, in general, be collapsed into one affine transformation.

That is the mathematical reason deep networks can build complex functions from simple pieces.

---

## 14. A Hand Calculation Challenge

Let

$$
A=\begin{bmatrix}1&2\\3&0\end{bmatrix},
\qquad
\mathbf x=\begin{bmatrix}4\\5\end{bmatrix}.
$$

Predict first.

Then compute:

$$
\begin{aligned}
x'_1&=1(4)+2(5)=14\\
x'_2&=3(4)+0(5)=12.
\end{aligned}
$$

Therefore

$$
A\mathbf x=\begin{bmatrix}14\\12\end{bmatrix}.
$$

Now ask yourself: **where did the 14 come from geometrically?**

It is the first coordinate of the combination

$$
4\cdot\text{column}_1(A)+5\cdot\text{column}_2(A).
$$

---

## 15. Scientist's Experiment

Compare these four transformations:

$$
I=\begin{bmatrix}1&0\\0&1\end{bmatrix},
\quad
S=\begin{bmatrix}2&0\\0&1\end{bmatrix},
\quad
R=\begin{bmatrix}0&-1\\1&0\end{bmatrix},
\quad
H=\begin{bmatrix}1&1\\0&1\end{bmatrix}.
$$

For each matrix, predict:

1. where $\mathbf e_1$ goes;
2. where $\mathbf e_2$ goes;
3. what happens to a unit square;
4. whether area is preserved;
5. whether lengths are preserved;
6. whether the transformation is invertible.

Then run the notebook and compare your predictions.

This is the habit we want throughout mathematics:

> **Predict → calculate → visualize → explain.**

---

## 16. Failure Modes and Misconceptions

### ❌ “Every matrix multiplication preserves distances.”
False. A general matrix can stretch, squash, shear, or collapse distances.

### ❌ “A zero determinant means every output is zero.”
False. It means at least one direction of information is lost.

### ❌ “$AB=BA$ because multiplication is multiplication.”
False for matrices in general. The order of transformations matters.

### ❌ “$W\mathbf x+\mathbf b$ is linear.”
It is affine unless $\mathbf b=0$.

### ❌ “Two linear layers are twice as powerful as one.”
Not by themselves. They collapse to one linear transformation.

### ❌ “The columns of a matrix are just data storage.”
They tell you where the basis directions go, so they reveal the transformation itself.

---

## 17. Machine Learning Connection

Now Chapter 3 and Chapter 4 fit together:

$$
XW
$$

can be read in two compatible ways.

### Algebraic view

Every output entry is a dot product.

### Geometric view

$W$ transforms the input coordinates into a new representation.

A neural layer therefore does not simply "calculate numbers." It **changes the representation** of the input.

For one example:

$$
\mathbf x\rightarrow W\mathbf x+\mathbf b.
$$

For a batch:

$$
X\rightarrow XW+\mathbf b.
$$

And after a nonlinear activation:

$$
\boxed{X\rightarrow XW+\mathbf b\rightarrow\sigma(XW+\mathbf b)}.
$$

That is the skeleton we will use for the first actual neural network.

---

## 18. Exercises

### Level A — intuition

1. Explain why the columns of a matrix determine its action on every 2-D vector.
2. Describe scaling, reflection, rotation and shear without using formulas.
3. Explain why a determinant of zero means information loss.

### Level B — calculation

4. Multiply a $2\times2$ matrix by a vector by hand.
5. Compute the determinant of three matrices.
6. Verify a composition $BA\mathbf x$ by applying $A$ first and $B$ second.

### Level C — coding

7. Plot a square before and after a transformation.
8. Plot the basis vectors before and after transformation.
9. Build a slider playground for $a,b,c,d$.

### Level D — deep learning

10. Prove that two linear layers collapse into one linear layer.
11. Explain exactly how the bias changes a linear map into an affine map.
12. Explain why nonlinear activation is the feature that prevents the whole network from collapsing into one matrix.

---

## 🏁 Mastery Gate

You are ready for Chapter 5 when you can:

- interpret a matrix as a transformation;
- use basis vectors to understand what the columns mean;
- identify scaling, reflection, rotation and shear;
- distinguish linear from affine maps;
- explain why matrix multiplication order matters;
- compute and interpret a determinant in 2-D;
- explain invertibility as the ability to undo a transformation;
- explain why a deep network needs nonlinear activation.

## 🔬 Research Bridge

The simple 2-D picture grows into:

- eigenvectors and invariant directions;
- singular values and conditioning;
- rank and information bottlenecks;
- learned representation spaces;
- equivariance and symmetry;
- linear operators in high-dimensional spaces.

Keep this research question:

> **Which transformations change the representation while preserving the information a learning system needs?**

---

## What We Discovered

1. A matrix is an instruction for transforming vectors, not merely a storage table.
2. The columns tell us where the basis directions go.
3. General matrices can scale, rotate, reflect, shear, mix, and collapse dimensions.
4. Matrix multiplication composes transformations, which explains non-commutativity.
5. Determinant measures area scaling in 2-D and detects collapse when it is zero.
6. $W\mathbf x+\mathbf b$ is affine, not purely linear.
7. Stacking linear layers without nonlinearities does not create a truly deeper class of functions.
8. A neural network becomes expressive when nonlinear transformations are inserted between learned affine maps.

The bridge to the next chapter is:

$$
\boxed{\mathbf z=W\mathbf x+\mathbf b\rightarrow\mathbf a=\sigma(\mathbf z)}.
$$

**Next: Chapter 05 — Meet the Smallest Neural Network.**
