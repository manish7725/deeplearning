# Chapter 02 — Numbers Become Vectors

## 🧭 Where we are

**Previous:** [Chapter 01 — What Does It Mean for a Machine to Learn?](<../Lecture 01 - What Does It Mean for a Machine to Learn/blog.md>)  
**Today:** Numbers → vectors → geometry → similarity → weighted combinations  
**Next:** [Chapter 03 — Matrices: The Spreadsheet of Mathematics](<../Lecture 03 - Matrices: The Spreadsheet of Mathematics/blog.md>)

> Chapter 01 taught us that learning changes parameters to reduce error. But a model cannot learn from a story such as “Alice is good at maths.” It needs numbers. Today we learn how numbers can be organized into **vectors** so that machines can calculate with them.

## 1. A vector is a little data box

Suppose we describe a student using two measurements:

```text
Alice → [9, 8]
Bob   → [8, 9]
Cara  → [3, 4]
```

The vector

$$\mathbf{x}=\begin{bmatrix}9\\8\end{bmatrix}$$

means “9 in the first feature and 8 in the second.” Order matters:

$$[9,8]\neq[8,9]$$

because position 1 and position 2 may mean different things.

### Three ways to see the same vector

| View | Meaning |
|---|---|
| List | ordered numbers |
| Point | a location in feature space |
| Arrow | a direction with a length |

This is one of the most important mental models in machine learning: **the same numbers can be interpreted algebraically and geometrically.**

## 2. Vectors create a map

For a two-feature example, `[9, 8]` can be drawn at coordinate $(9,8)$. Once objects become points, we can ask mathematical questions:

- Which examples are close?
- Which direction is a feature changing?
- How large is a vector?
- How strongly do two vectors point in related directions?

For Alice $A=(9,8)$ and Bob $B=(8,9)$:

$$d(A,B)=\sqrt{(9-8)^2+(8-9)^2}=\sqrt2$$

For Alice and Cara $C=(3,4)$:

$$d(A,C)=\sqrt{(9-3)^2+(8-4)^2}=\sqrt{52}\approx7.21$$

So, **under this representation**, Alice is much closer to Bob.

> Important: distance is only as meaningful as the representation and units we chose.

## 3. Length: the norm

For

$$\mathbf v=\begin{bmatrix}3\\2\end{bmatrix}$$

its length is

$$\|\mathbf v\|=\sqrt{3^2+2^2}=\sqrt{13}\approx3.606.$$

For a vector with $n$ coordinates:

$$\|\mathbf x\|_2=\sqrt{\sum_{i=1}^{n}x_i^2}.$$

The subscript 2 means this is the familiar Euclidean length.

## 4. Add arrows, not just numbers

Let

$$\mathbf a=\begin{bmatrix}2\\1\end{bmatrix},\qquad\mathbf b=\begin{bmatrix}1\\3\end{bmatrix}.$$

Then

$$\mathbf a+\mathbf b=\begin{bmatrix}3\\4\end{bmatrix}.$$

Imagine walking 2 steps right and 1 up, then 1 right and 3 up. Your total movement is 3 right and 4 up.

Scalar multiplication stretches or flips an arrow:

$$2\mathbf x=\begin{bmatrix}4\\6\end{bmatrix},\qquad-\mathbf x=\begin{bmatrix}-2\\-3\end{bmatrix}.$$

## 5. The dot product: the weighted-sum engine

Let

$$\mathbf x=\begin{bmatrix}2\\3\end{bmatrix},\qquad\mathbf w=\begin{bmatrix}4\\5\end{bmatrix}.$$

Then

$$\mathbf x\cdot\mathbf w=2(4)+3(5)=23.$$

In $n$ dimensions:

$$\mathbf x\cdot\mathbf w=\sum_{i=1}^{n}x_iw_i.$$

This is already the heart of a neuron. Each input is multiplied by a weight, and the results are added:

$$z=\mathbf w\cdot\mathbf x+b.$$

We will build the neuron explicitly in a later chapter; for now, recognize the connection.

### A geometric surprise

The dot product also tells us about direction:

$$\mathbf x\cdot\mathbf w=\|\mathbf x\|\,\|\mathbf w\|\cos\theta.$$

So the same operation can be understood as a **weighted sum** or as a **measure of directional alignment**.

## 6. Dimension versus shape

A vector such as

```python
x = np.array([2, 3, 4])
```

has three coordinates and shape `(3,)`.

Now put three vectors together:

```python
X = np.array([
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10],
])
```

Its shape is `(3, 3)`: three examples, three features.

This is our doorway into matrices.

> **Dimension tells us how many coordinates one object has. Shape tells us how the data is arranged.**

## 7. Representation can help — or lie

Suppose we encode colors as:

```text
red = 1
blue = 2
orange = 3
```

The numbers suggest an ordering and distances that may not exist in the real concept. A better representation might use several features, or a learned embedding.

Therefore:

$$\boxed{\text{Good learning depends on useful representations.}}$$

This idea eventually leads to embeddings and representation learning.

## 8. Interactive mathematical playground

The branch is being built toward interactive math. For this lesson, the playground should let you drag the endpoints of two vectors and immediately see:

1. both arrows;
2. their lengths;
3. their sum;
4. the angle between them;
5. the dot product;
6. the effect of changing one coordinate.

A useful animation is:

```text
x ──multiply by w₁──┐
                    ├── add ──→ x · w
y ──multiply by w₂──┘
```

The static equations in this chapter remain the source of truth even when JavaScript is unavailable.

## 9. Scientist challenge

Choose three objects near you. Pick three measurable features and represent each object as a vector.

Then predict before calculating:

1. Which pair will be closest?
2. Which feature contributes most to the distance?
3. What happens if one feature is measured in centimetres and another in metres?
4. What happens if you multiply every coordinate by 10?

That last question introduces an important research habit: **changing the representation can change the geometry.**

## 10. NumPy checkpoint

```python
import numpy as np

x = np.array([2.0, 3.0])
w = np.array([4.0, 5.0])

print("addition:", x + w)
print("double x:", 2 * x)
print("dot product:", x @ w)
print("length of x:", np.linalg.norm(x))
```

Predict the outputs before running it. Then change one number and explain exactly which outputs should change.

## 11. Common mistakes

**Mistake 1: A vector is just an ordinary list.**  
A vector is an ordered mathematical object; its coordinates have meaning.

**Mistake 2: More dimensions mean a better representation.**  
Not necessarily. Extra coordinates can be noisy, redundant, or badly scaled.

**Mistake 3: Distance always means similarity.**  
Only if the representation and metric make that interpretation sensible.

**Mistake 4: Dot product is only a formula.**  
It is both a weighted sum and a geometric alignment measurement.

## 12. Exercises

1. Calculate the norm of $[6,8]$.
2. Calculate $[2,5]+[-1,3]$.
3. Calculate $[2,5]\cdot[4,-1]$.
4. Find the distance between $(1,2)$ and $(4,6)$.
5. Explain why `[height, weight]` and `[weight, height]` are different representations.
6. Create two vectors with dot product zero. What does that tell you geometrically?

### Mini-project
Build a tiny “object map” in Python. Represent at least five objects with two features, plot them, and experiment with the definition of distance.

### Research extension
Repeat the mini-project after scaling one feature by 100. Does the nearest-neighbour relationship change? Why?

## What to remember

$$\boxed{\text{numbers}\rightarrow\text{vectors}\rightarrow\text{geometry}\rightarrow\text{operations}}$$

A vector lets a machine represent one example mathematically. Many vectors naturally form a matrix.

**Next: matrices — the spreadsheet of mathematics.**
