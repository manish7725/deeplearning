# Lecture 05 — Meet the Smallest Neural Network

> **The Big Question:** How do a few numbers become a machine that can learn?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2005%20-%20Meet%20the%20Smallest%20Neural%20Network/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Chapter 4 showed that a matrix transforms a vector. We also discovered that $W\mathbf x+\mathbf b$ is an affine transformation and that nonlinear functions are needed to prevent stacked layers from collapsing into one affine map.

**Today:** We turn that calculation into the smallest trainable neural network and identify every piece of it.

**Next:** We ask why the activation function is necessary and what changes when a neuron becomes nonlinear.

---

## 1. Start With One Input

Suppose a student studies for $x$ hours and we want to predict a score.

Our first model can be

$$
\hat y=wx.
$$

If

$$
x=3,\qquad w=2,
$$

then

$$
\hat y=2(3)=6.
$$

There is nothing mysterious here. It is simply a function.

The number $w$ controls how strongly the input affects the output.

- $w>0$: increasing $x$ increases the prediction;
- $w<0$: increasing $x$ decreases the prediction;
- $w=0$: the input contributes nothing.

But remember a subtle point from Chapter 4:

> **A learned weight describes how the model uses a feature; it is not automatically a real-world causal explanation.**

---

## 2. Add the Bias

Most useful lines do not have to pass through the origin.

So add a bias:

$$
\boxed{\hat y=wx+b}.
$$

For

$$
w=2,\qquad x=3,\qquad b=1,
$$

we get

$$
\hat y=2(3)+1=7.
$$

Geometrically:

- $w$ controls the **slope**;
- $b$ controls the **vertical shift**.

So the smallest neuron before activation is just a line.

---

## 3. Why This Is Already a Neuron

Chapter 2 gave us the dot product.

Chapter 3 gave us matrix multiplication.

Chapter 4 gave us transformations.

Now put them together.

For several inputs,

$$
\mathbf x=
\begin{bmatrix}
2\\3\\4
\end{bmatrix}
$$

and weights

$$
\mathbf w=
\begin{bmatrix}
4\\5\\2
\end{bmatrix},
\qquad b=1,
$$

the neuron computes

$$
z=\mathbf w^T\mathbf x+b.
$$

Step by step:

$$
z=4(2)+5(3)+2(4)+1
$$

$$
z=8+15+8+1=32.
$$

That is the complete pre-activation calculation.

> 💡 **A neuron is not a tiny brain.**
>
> An artificial neuron is a mathematical function that combines inputs using parameters and produces an output.

---

## 4. Read the Neuron as a Flow of Contributions

Think of each input as making a contribution:

```text
x₁ ──× w₁ ──┐
             │
x₂ ──× w₂ ──┼── add ── + b ── z
             │
x₃ ──× w₃ ──┘
```

The formula is

$$
\boxed{z=\sum_{i=1}^{d}w_ix_i+b}.
$$

Nothing is hidden:

1. multiply each input by its weight;
2. add the contributions;
3. add the bias;
4. call the result $z$.

This is the same weighted sum we already saw in the dot product.

---

## 5. Negative Weights Are Allowed

Take

$$
\mathbf x=
\begin{bmatrix}2\\3\end{bmatrix},
\qquad
\mathbf w=
\begin{bmatrix}4\\-5\end{bmatrix},
\qquad b=1.
$$

Then

$$
z=4(2)-5(3)+1=-6.
$$

The second input pulls the weighted sum downward.

So a neuron is not just adding features. It is adding **signed contributions**.

---

## 6. From One Neuron to a Layer

One neuron gives one number.

Suppose we want three numbers from the same input.

Use three neurons.

Let

$$
W=
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix},
\qquad
\mathbf x=\begin{bmatrix}2\\3\end{bmatrix}.
$$

Then

$$
W\mathbf x=
\begin{bmatrix}
1(2)+2(3)\\
3(2)+4(3)\\
5(2)+6(3)
\end{bmatrix}
=
\begin{bmatrix}8\\18\\28\end{bmatrix}.
$$

Each row of $W$ is one neuron's weight vector.

This is the matrix lesson becoming a neural-network lesson:

> **One matrix multiplication can perform many neuron calculations simultaneously.**

Add biases:

$$
\mathbf z=W\mathbf x+\mathbf b.
$$

Then apply an activation:

$$
\mathbf a=\sigma(\mathbf z).
$$

A neural-network layer therefore has the basic structure

$$
\boxed{\mathbf x\rightarrow W\mathbf x+\mathbf b\rightarrow\sigma(\cdot)}.
$$

---

## 7. Count the Parameters

Suppose a neuron has $d$ inputs.

It needs:

- $d$ weights;
- 1 bias.

So

$$
\boxed{d+1}
$$

trainable parameters.

For a layer with $d$ inputs and $m$ neurons:

$$
W\in\mathbb R^{m\times d},
\qquad
\mathbf b\in\mathbb R^m.
$$

The number of parameters is

$$
md+m=m(d+1).
$$

For example, 4 inputs and 3 neurons require

$$
3(4)+3=15
$$

parameters.

Parameter counting matters because parameters consume memory and computation.

---

## 8. Keep Data and Parameters Separate

This distinction is essential.

| Symbol | Meaning | Learned? |
|---|---|---|
| $\mathbf x$ | input example | No, not by the model |
| $W$ | weights | Yes |
| $\mathbf b$ | biases | Yes |
| $\mathbf z$ | weighted sum | Computed |
| $\mathbf a$ | activated output | Computed |
| $y$ | true target | Comes from data |
| $\hat y$ | prediction | Computed |

Training does not normally alter the training examples. It changes $W$ and $\mathbf b$ so that the model's predictions improve.

This is the first place where the word **parameter** should become concrete:

> **A parameter is a number the learning algorithm is allowed to change.**

---

## 9. A Batch of Examples

One example at a time is useful for understanding, but training usually uses many examples together.

Let

$$
X=
\begin{bmatrix}
1&2\\
2&3\\
3&4
\end{bmatrix}
$$

and

$$
\mathbf w=\begin{bmatrix}2\\3\end{bmatrix},
\qquad b=1.
$$

Then

$$
X\mathbf w=
\begin{bmatrix}
8\\13\\18
\end{bmatrix}
$$

and

$$
\hat{\mathbf y}
=X\mathbf w+b
=
\begin{bmatrix}9\\14\\19\end{bmatrix}.
$$

The bias is broadcast to each example.

This is **vectorized computation**: the mathematics describes the whole batch, while the numerical library performs optimized loops underneath.

---

## 10. Prediction Is Not Learning

Suppose

$$
w=0,\qquad b=0.
$$

Then every input gives

$$
\hat y=0.
$$

The neuron can calculate perfectly according to its current parameters.

But it has no idea whether zero is a good prediction.

That requires a second ingredient: a **loss function**.

For a target $y$ and prediction $\hat y$, Chapter 1 introduced squared error:

$$
L=(\hat y-y)^2.
$$

Now our machine has two separate processes:

### Forward computation

$$
\mathbf x\rightarrow W\mathbf x+\mathbf b\rightarrow\hat y
$$

### Learning

$$
\hat y,y\rightarrow L\rightarrow\text{change }W,\mathbf b
$$

This distinction is fundamental.

> 🧠 **A model can predict without learning. Learning requires a way to evaluate the prediction and update the parameters.**

---

## 11. What Is Actually Trainable?

For a single neuron,

$$
\hat y=w_1x_1+w_2x_2+b.
$$

The input values $x_1,x_2$ are given for the example.

The model can change

$$
\boxed{w_1,w_2,b}.
$$

So the training process searches over parameter space:

```text
(w₁, w₂, b)
     │
     ▼
 prediction
     │
     ▼
    loss
     │
     ▼
 better parameters?
```

This is the bridge from Chapter 1's line-fitting problem to a neural network.

The old model had two knobs, $w$ and $b$.

A neuron with two inputs has three knobs, $w_1,w_2,b$.

A layer with millions of parameters has millions of knobs.

The learning problem is the same idea at larger scale.

---

## 12. Build the Neuron Yourself in NumPy

```python
import numpy as np

x = np.array([2.0, 3.0, 4.0])
w = np.array([4.0, 5.0, 2.0])
b = 1.0

z = w @ x + b
print(z)  # 32.0
```

The `@` operator means matrix multiplication. For two 1-D vectors here, it is the dot product.

The important point is not the library call.

It is the mathematics:

**multiply → add → produce a number.**

---

## 13. The Same Mathematics in PyTorch

```python
import torch

x = torch.tensor([2., 3., 4.])
w = torch.tensor([4., 5., 2.])
b = torch.tensor(1.)

z = w @ x + b
print(z)
```

PyTorch provides automatic differentiation, optimizers, GPU support and neural-network modules, but the underlying arithmetic is unchanged.

> **Understand the arithmetic first. Use PyTorch second.**

---

## 14. The Smallest Trainable Network

The smallest useful supervised network needs at least:

1. inputs;
2. parameters;
3. a forward rule;
4. a loss;
5. a parameter-update rule.

For a single-neuron regression model:

$$
\hat y=wx+b
$$

with loss

$$
L=(\hat y-y)^2.
$$

Substitute the prediction:

$$
L=(wx+b-y)^2.
$$

Now the loss depends directly on the parameters $w$ and $b$.

That means there is something we can optimize.

Chapter 1 gave us the idea of gradient descent. Later chapters derive the gradients carefully. The architecture itself is already here.

---

## 15. Why Add an Activation Function?

At this stage the neuron still computes only

$$
z=\mathbf w^T\mathbf x+b.
$$

So it is an affine function.

If we stack ten such affine layers with no nonlinear operation between them, the whole network can be rewritten as one affine function.

Therefore something new must happen between layers:

$$
\boxed{\mathbf a=\sigma(\mathbf z)}.
$$

The next chapter studies this carefully.

Do not memorize "ReLU is common" and move on. The interesting mathematical question is:

> **What can a nonlinear function do that a collection of affine functions cannot?**

---

## 16. Interactive Neuron Playground

The notebook should let you change one parameter at a time.

Try:

1. keep $w$ fixed and increase $b$;
2. keep $b$ fixed and increase $w$;
3. set one weight to zero;
4. make one weight negative;
5. change one input while holding everything else fixed.

For a browser or notebook visualization, show the pipeline:

$$
x_i\xrightarrow{\times w_i}w_ix_i
\xrightarrow{\text{sum}}\mathbf w^T\mathbf x
\xrightarrow{+b}z
\xrightarrow{\sigma}\mathbf a.
$$

The student should be able to see that changing a parameter changes a numerical contribution before it changes the final output.

---

## 17. Scientist's Experiment

Start with

$$
\hat y=3x-2.
$$

Calculate the predictions for

$$
x=0,1,2,3.
$$

Then change only the weight:

$$
w=1,
$$

keeping $b=-2$.

Ask:

- What changed geometrically?
- What stayed fixed?

Now restore $w=3$ and change only the bias:

$$
b=4.
$$

Ask the same questions.

You should discover experimentally:

- changing $w$ changes the slope;
- changing $b$ shifts the line.

Then repeat the experiment with two input features and watch each weight control one direction of the input space.

---

## 18. Common Misconceptions

### ❌ “A neuron is automatically learning.”
No. A neuron is first a function. Learning requires a loss and parameter updates.

### ❌ “A large weight means the feature is important in reality.”
Not necessarily. Scaling and model structure matter.

### ❌ “Matrix multiplication is separate from neural networks.”
Dense layers are fundamentally matrix multiplication plus bias, followed by an activation in the usual feed-forward formulation.

### ❌ “The bias is just another weight.”
It is a parameter, but unlike an ordinary feature weight it does not multiply an input feature.

### ❌ “More layers automatically mean more expressive power.”
Not if all layers are affine. Without nonlinearity, they collapse into one affine transformation.

---

## 19. Failure Modes

A tiny network already fails in instructive ways.

| Failure | What it means |
|---|---|
| All weights zero | The model ignores every input. |
| Bias zero when an offset is needed | The model is forced through the origin. |
| Wrong feature order | Parameters multiply the wrong inputs. |
| Wrong matrix shape | A neuron or layer cannot pair inputs with weights. |
| No activation in a deep stack | The entire stack may collapse to one affine map. |
| Prediction computed but no loss | The model has no numerical signal telling it whether it is wrong. |

> ⚠️ **A useful debugging habit:** write the shape and meaning of every tensor before writing the code.

---

## 20. Exercises

### Level A — Calculate

1. Compute $z$ for $x=[2,5]$, $w=[3,-1]$, $b=4$.
2. Compute the outputs of a 3-neuron layer for a 2-element input.
3. Count the parameters in a neuron with 7 inputs.

### Level B — Explain

4. Explain the difference between a weight and a bias using a graph.
5. Why can three neurons be represented by one matrix multiplication?
6. Why is prediction different from learning?

### Level C — Investigate

7. Build a neuron with 5 inputs and count its parameters.
8. Change exactly one parameter and measure the output change.
9. Implement a batch of 100 examples using matrix multiplication.

### Level D — Deep Learning Bridge

10. Write the forward pass for one dense layer as one equation.
11. Explain why two affine layers without activation can collapse into one affine map.
12. Explain why a nonlinear activation changes that conclusion.

---

## 🏁 Mastery Gate

Move to Chapter 6 only if you can:

- calculate $\mathbf w^T\mathbf x+b$ by hand;
- explain weight and bias geometrically;
- express multiple neurons as $W\mathbf x+\mathbf b$;
- count parameters in a dense layer;
- explain the difference between data, parameters and predictions;
- implement the neuron in NumPy;
- explain why prediction is not yet learning;
- explain why a nonlinear activation will be necessary.

## 🔬 Research Bridge

This tiny neuron grows naturally into questions about:

- parameter efficiency;
- initialization;
- optimization landscapes;
- feature representations;
- expressivity and universal approximation;
- sparse and mixture-of-experts layers;
- hardware-efficient matrix multiplication.

The research question to keep:

> **How can a huge function be built from simple parameterized operations without losing trainability?**

---

## What We Discovered

1. A neuron begins with a weighted sum and a bias.
2. $\mathbf w^T\mathbf x$ is the same dot product we learned earlier.
3. A layer is many neurons packed into one matrix multiplication.
4. The number of parameters follows directly from the matrix shape.
5. Inputs are data; weights and biases are trainable parameters.
6. A forward pass produces predictions but does not itself learn.
7. Learning begins when predictions are compared with targets through a loss and the parameters are updated.
8. Stacking affine transformations without nonlinearity does not create a fundamentally nonlinear model.

The central equation is

$$
\boxed{\mathbf z=W\mathbf x+\mathbf b}
$$

and the full neural-layer pattern is

$$
\boxed{\mathbf x\rightarrow W\mathbf x+\mathbf b\rightarrow\sigma(\mathbf z)}.
$$

**Next: Chapter 06 — Why Does a Neuron Need an Activation Function?**
