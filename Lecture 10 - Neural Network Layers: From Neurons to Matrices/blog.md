# Lecture 10 — Neural Network Layers: From Neurons to Matrices

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 Open the notebook on GitHub](https://github.com/manish7725/deeplearning/blob/main/Lecture%2010%20-%20Neural%20Network%20Layers:%20From%20Neurons%20to%20Matrices/notebook.ipynb)** · **[▶ Open in Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2010%20-%20Neural%20Network%20Layers:%20From%20Neurons%20to%20Matrices/notebook.ipynb)**

## 🧭 Where this lesson fits

**Previous lesson:** Backpropagation showed how gradients travel through one computational graph.

**Today:** We put many neurons together and discover that matrix multiplication is simply many neuron calculations performed at once.

**Next lesson:** Why nonlinear activation functions are necessary for deep networks to learn genuinely complex relationships.

---

## 1. One neuron was useful — but small

A neuron computes

$$
z=w_1x_1+w_2x_2+b$$

For two inputs, it takes two numbers, weights them, adds them, and shifts the result.

That is enough for a simple decision or prediction.

But suppose an image has 784 input numbers.

Writing

$$
w_1x_1+w_2x_2+\cdots+w_{784}x_{784}+b
$$

works mathematically, but becomes hard to reason about.

We need a better notation.

---

## 2. A neuron is a dot product

Put the inputs into a vector:

$$
\mathbf{x}
=
\begin{bmatrix}
 x_1\\x_2
\end{bmatrix}
$$

and the weights into another:

$$
\mathbf{w}
=
\begin{bmatrix}
 w_1\\w_2
\end{bmatrix}
$$

Then the weighted sum becomes

$$
\mathbf{w}^T\mathbf{x}
=w_1x_1+w_2x_2
$$

So the neuron is

$$
z=\mathbf{w}^T\mathbf{x}+b
$$

The dot product is not a mysterious deep-learning operation. It is simply a compact way to say:

> multiply matching entries and add the results.

---

## 3. Two neurons

Now build two neurons that look at the same input.

Neuron 1:

$$
z_1=w_{11}x_1+w_{12}x_2+b_1
$$

Neuron 2:

$$
z_2=w_{21}x_1+w_{22}x_2+b_2
$$

Put both sets of weights into one matrix:

$$
W=
\begin{bmatrix}
w_{11}&w_{12}\\
w_{21}&w_{22}
\end{bmatrix}
$$

and the biases into a vector:

$$
\mathbf b=
\begin{bmatrix}b_1\\b_2\end{bmatrix}
$$

Now both neurons can be calculated together:

$$
\mathbf z=W\mathbf x+\mathbf b
$$

This is the central equation for a dense layer.

---

## 4. Do the matrix multiplication by hand

Let

$$
W=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix},
\qquad
\mathbf x=
\begin{bmatrix}
2\\1
\end{bmatrix},
\qquad
\mathbf b=
\begin{bmatrix}
1\\-1
\end{bmatrix}
$$

First multiply:

$$
W\mathbf x
=
\begin{bmatrix}
1(2)+2(1)\\
3(2)+4(1)
\end{bmatrix}
=
\begin{bmatrix}
4\\10
\end{bmatrix}
$$

Then add the bias:

$$
\mathbf z
=
\begin{bmatrix}4\\10\end{bmatrix}
+
\begin{bmatrix}1\\-1\end{bmatrix}
=
\begin{bmatrix}5\\9\end{bmatrix}
$$

The first row is neuron 1. The second row is neuron 2.

> 💡 **A matrix is a bundle of neurons.**

Each row contains the weights for one neuron.

---

## 5. A layer is many neurons sharing an input

If there are $n$ inputs and $m$ neurons, then

$$
W\in\mathbb{R}^{m\times n}
$$

The input has shape

$$
\mathbf x\in\mathbb{R}^{n}
$$

and the output has shape

$$
\mathbf z\in\mathbb{R}^{m}
$$

The multiplication is legal because the inner dimensions match:

$$
(m\times n)(n\times1)=(m\times1)
$$

This shape rule is one of the most useful habits in machine learning.

Before calculating numbers, ask:

> **What is the shape of every object?**

---

## 6. A batch of examples

So far we processed one example.

Training normally uses many examples together.

Suppose four examples are stored as rows:

$$
X=
\begin{bmatrix}
 x^{(1)T}\\
 x^{(2)T}\\
 x^{(3)T}\\
 x^{(4)T}
\end{bmatrix}
$$

If $X$ has shape $4\times n$ and $W$ has shape $m\times n$, it is often convenient to store weights in the transposed orientation and write

$$
Z=XW^T+b
$$

Now four examples pass through all neurons in one matrix operation.

The hardware loves this because matrix multiplication maps efficiently to GPUs and other accelerators.

---

## 7. From one layer to two layers

Suppose the first layer computes

$$
\mathbf h=W_1\mathbf x+\mathbf b_1
$$

and the second computes

$$
\mathbf z=W_2\mathbf h+\mathbf b_2
$$

Then

$$
\mathbf x
\rightarrow
\mathbf h
\rightarrow
\mathbf z
$$

The output of one layer becomes the input of the next.

Without a nonlinear activation, however, something surprising happens.

---

## 8. Two linear layers collapse into one

Ignore biases for a moment:

$$
\mathbf z=W_2(W_1\mathbf x)
$$

Associativity gives

$$
\mathbf z=(W_2W_1)\mathbf x
$$

So two linear transformations are equivalent to one linear transformation.

Three linear layers are still just one linear transformation.

> ⚠️ This is a crucial failure mode in our mental model. **Depth alone does not automatically create a more powerful function.**

Something else is required.

That missing ingredient is the subject of the next lesson.

---

## 9. Why the matrix view matters

The matrix formulation gives us three huge benefits.

### Compact mathematics

Instead of writing every neuron separately, one equation describes the whole layer.

### Efficient computation

Libraries can send large matrix multiplications to optimized CPU/GPU kernels.

### Clean gradients

Backpropagation can also be expressed with matrix operations.

For

$$
Z=XW^T+b
$$

and a loss $L$, the backward pass produces gradients for the whole matrix $W$, not one weight at a time.

---

## 10. A layer in Python

```python
import numpy as np

W = np.array([[1., 2.],
              [3., 4.]])
b = np.array([1., -1.])
x = np.array([2., 1.])

z = W @ x + b
print(z)
```

The `@` operator means matrix multiplication.

It corresponds directly to

$$
W\mathbf x
$$

Seeing the same equation in mathematics and code is an important skill.

---

## 11. The same layer in PyTorch

```python
import torch

layer = torch.nn.Linear(2, 2)
x = torch.tensor([2.0, 1.0])

z = layer(x)
print(z)
```

`Linear(2, 2)` means:

- 2 input features;
- 2 output neurons;
- one weight for every input-output connection;
- one bias for every output neuron.

The framework is packaging the same mathematics we just wrote by hand.

---

## Think Like a Scientist 🧠

Take

$$
W=
\begin{bmatrix}
2&-1&0\\
1&3&2
\end{bmatrix}
$$

and

$$
\mathbf x=
\begin{bmatrix}1\\2\\3\end{bmatrix}
$$

with

$$
\mathbf b=
\begin{bmatrix}1\\-2\end{bmatrix}.
$$

Calculate $W\mathbf x+\mathbf b$ by hand.

Then state the shape of every object before doing the arithmetic.

Finally, explain why a $3\times2$ matrix would be wrong here.

---

## What you should remember

> **A dense neural-network layer is many neurons written as one matrix equation: $\mathbf z=W\mathbf x+\mathbf b$.**

The structure is now visible:

$$
\text{inputs}
\rightarrow
\text{matrix multiplication}
\rightarrow
\text{bias}
\rightarrow
\text{output}
$$

But stacked linear layers collapse into one linear transformation.

> **Next: activation functions — the small nonlinear step that makes deep networks genuinely deep.**

---

# 📚 Go Deeper

Try changing one element of $W$ and observe which output neuron changes. This makes the meaning of an individual weight concrete: a weight controls one specific connection from one input feature to one neuron.
