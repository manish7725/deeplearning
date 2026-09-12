# Lecture 23 — Backpropagation: Sending the Error Backward

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

**[📓 GitHub notebook](https://github.com/manish7725/deeplearning/blob/main/Lecture%2023%20-%20Backpropagation%3A%20Sending%20the%20Error%20Backward/notebook.ipynb)** · **[▶ Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2023%20-%20Backpropagation%3A%20Sending%20the%20Error%20Backward/notebook.ipynb)**

The blog is the textbook; the notebook is the laboratory. Work every derivative by hand before using automatic differentiation.

## 🧭 Where this lesson fits

**Previous:** Lecture 22 — Why Depth Makes Neural Networks Powerful.

**Today:** We answer the central learning question: how does the final error tell every earlier parameter what to change?

**Next:** Lecture 24 — Meet the Smallest Neural Network.

The key bridge is:

> **Backpropagation is repeated use of the chain rule to move information about the loss backward through the computation graph.**

---

## 1. Forward pass: the easy direction

Suppose a tiny network computes

$$z=wx+b$$

then

$$\hat y=z$$

and finally the squared loss

$$L=(\hat y-y)^2.$$

The forward pass is:

```text
x → multiply by w → add b → prediction → compare with y → loss
```

For example, let

$$x=2,\quad w=3,\quad b=1,\quad y=10.$$

Then

$$z=3(2)+1=7$$

and

$$L=(7-10)^2=9.$$

We know the model is wrong.

But the real learning question is:

> **Should $w$ increase? Should $b$ increase? By how much?**

---

## 2. A derivative answers the question

The quantity we want is

$$\frac{\partial L}{\partial w}.$$

Read that as:

> “How much does the loss change if I change the weight a tiny amount?”

That is exactly the information an optimizer needs.

---

## 3. The dependency chain

The loss depends on $w$ through several intermediate quantities:

$$w\rightarrow z\rightarrow \hat y\rightarrow L.$$

So we use the chain rule:

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
\frac{\partial z}{\partial w}
}
$$

The derivatives multiply because a small change travels through the whole chain.

---

## 4. Derive it one step at a time

Our loss is

$$L=(\hat y-y)^2.$$

Therefore

$$\frac{\partial L}{\partial\hat y}=2(\hat y-y).$$

Because $\hat y=z$,

$$\frac{\partial\hat y}{\partial z}=1.$$

Because $z=wx+b$,

$$\frac{\partial z}{\partial w}=x.$$

Multiply them:

$$\boxed{\frac{\partial L}{\partial w}=2(\hat y-y)x}.$$

For our numbers,

$$\hat y-y=7-10=-3$$

and $x=2$, so

$$\frac{\partial L}{\partial w}=2(-3)(2)=-12.$$

The gradient is negative.

So increasing $w$ slightly would decrease the loss at this point.

---

## 5. What about the bias?

Since

$$\frac{\partial z}{\partial b}=1,$$

we get

$$\boxed{\frac{\partial L}{\partial b}=2(\hat y-y)}.$$

For our example:

$$\frac{\partial L}{\partial b}=-6.$$

The model predicted too low, so increasing both $w$ and $b$ is locally helpful.

---

## 6. Gradient descent closes the loop

Gradient descent uses the derivative to update a parameter:

$$\boxed{\theta_{new}=\theta_{old}-\eta\frac{\partial L}{\partial\theta}}$$

where $\eta$ is the learning rate.

For $w$, with $\eta=0.1$:

$$w_{new}=3-0.1(-12)=4.2.$$

For $b$:

$$b_{new}=1-0.1(-6)=1.6.$$

Now do another forward pass:

$$\hat y=4.2(2)+1.6=10.$$

and therefore

$$L=(10-10)^2=0.$$

The entire learning loop is:

```text
forward → loss → derivatives → update → forward again
```

---

## 7. Hidden layers can learn too

Now add a hidden unit:

$$z_1=w_1x+b_1$$

$$h=f(z_1)$$

$$\hat y=w_2h+b_2.$$

The dependency becomes

$$w_1\rightarrow z_1\rightarrow h\rightarrow\hat y\rightarrow L.$$

So

$$
\boxed{
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial h}
\frac{\partial h}{\partial z_1}
\frac{\partial z_1}{\partial w_1}
}
$$

That is backpropagation.

Nothing magical happened. We simply kept applying the chain rule.

---

## 8. Work the hidden-layer example

Let

$$x=2,\quad w_1=2,\quad b_1=1,\quad w_2=3,\quad b_2=0,\quad y=18.$$

Use ReLU.

### Forward pass

$$z_1=2(2)+1=5$$

$$h=\operatorname{ReLU}(5)=5$$

$$\hat y=3(5)=15$$

Therefore

$$L=(15-18)^2=9.$$

### Step 1: loss to prediction

$$\frac{\partial L}{\partial\hat y}=2(15-18)=-6.$$

### Step 2: prediction to hidden activation

$$\frac{\partial\hat y}{\partial h}=3.$$

So

$$\frac{\partial L}{\partial h}=(-6)(3)=-18.$$

### Step 3: ReLU derivative

Because $z_1=5>0$,

$$\frac{\partial h}{\partial z_1}=1.$$

Thus

$$\frac{\partial L}{\partial z_1}=-18.$$

### Step 4: pre-activation to first weight

$$\frac{\partial z_1}{\partial w_1}=x=2.$$

Therefore

$$\boxed{\frac{\partial L}{\partial w_1}=-36}.$$

The error has travelled all the way from the output back to the first weight.

---

## 9. The gradient is information flowing backward

A useful mental picture is:

```text
FORWARD
x ──→ z₁ ──→ h ──→ ŷ ──→ L

BACKWARD
x ←──      ←──   ←──   ←──
       gradients of L
```

The forward pass computes values.

The backward pass computes sensitivities.

Those sensitivities tell us how strongly each parameter affects the final loss.

---

## 10. Why the chain rule multiplies

Suppose a tiny change $dw$ causes

$$dz\approx\frac{\partial z}{\partial w}dw.$$

That change causes

$$dh\approx\frac{\partial h}{\partial z}dz.$$

And then

$$dL\approx\frac{\partial L}{\partial h}dh.$$

Substituting gives

$$
 dL
\approx
\frac{\partial L}{\partial h}
\frac{\partial h}{\partial z}
\frac{\partial z}{\partial w}
 dw.
$$

So the overall sensitivity is the product

$$\boxed{\frac{\partial L}{\partial w}=\frac{\partial L}{\partial h}\frac{\partial h}{\partial z}\frac{\partial z}{\partial w}}.$$

The chain rule is bookkeeping for how small changes travel through a sequence of functions.

---

## 11. A computation graph makes it visible

For

$$L=(3\,\operatorname{ReLU}(2x+1)-y)^2$$

think of the graph as:

```text
x
│
× 2
│
+ 1
│
z
│
ReLU
│
h
│
× 3
│
ŷ
│
compare with y
│
square
│
L
```

Every arrow has a local derivative.

Backpropagation walks these arrows backward and multiplies the local derivatives.

---

## 12. What automatic differentiation really does

PyTorch can calculate these derivatives automatically:

```python
import torch

x = torch.tensor(2.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)

pred = w * x + b
loss = (pred - y) ** 2
loss.backward()

print('prediction:', pred.item())
print('loss:', loss.item())
print('dL/dw:', w.grad.item())
print('dL/db:', b.grad.item())
```

The library is not replacing calculus.

It is performing the calculus for the computation graph we created.

A good rule is:

> **Derive tiny examples by hand; use automatic differentiation for large graphs.**

---

## 13. Why gradients can become tiny or huge

In a deep network, a gradient can contain many factors:

$$
\frac{\partial L}{\partial w_1}
=
\frac{\partial L}{\partial a_L}
\frac{\partial a_L}{\partial a_{L-1}}
\cdots
\frac{\partial a_2}{\partial a_1}
\frac{\partial a_1}{\partial w_1}.
$$

If many factors have magnitude below $1$, their product can become extremely small.

This is the **vanishing-gradient** problem.

If many factors have magnitude above $1$, the product can become very large.

That is the **exploding-gradient** problem.

This is one reason initialization, activation functions, normalization, residual connections, and optimizer design matter in deep learning.

---

## 14. Backpropagation versus gradient descent

These are related but different.

### Backpropagation
Computes the gradients:

$$\nabla_\theta L.$$

### Gradient descent
Uses those gradients to update parameters:

$$\theta\leftarrow\theta-\eta\nabla_\theta L.$$

So:

```text
backpropagation → computes how the loss depends on parameters
optimizer       → uses those gradients to change parameters
```

Backpropagation is not the optimizer.

---

## 15. One complete learning step

### Step 1 — Forward

$$\mathbf x\rightarrow\text{prediction}$$

### Step 2 — Loss

$$\text{prediction},\mathbf y\rightarrow L$$

### Step 3 — Backward

$$L\rightarrow\nabla_\theta L$$

### Step 4 — Update

$$\theta\leftarrow\theta-\eta\nabla_\theta L$$

### Step 5 — Repeat

Training is this cycle repeated many times.

> **There is no separate learning spell. Repeated parameter updates are the learning process.**

---

## 🎮 Interactive gradient playground

Start with

$$\hat y=wx+b$$

and a target $y$.

Plot the loss as a function of $w$.

Then show:

```text
current w
   ↓
current loss
   ↓
gradient
   ↓
next w
```

The most useful animation is the parameter moving down the loss curve.

Then extend it to $(w,b)$ and visualize the loss surface.

---

## ⚠️ Common misconceptions

### “Backpropagation changes the weights.”
Backpropagation computes gradients. The optimizer performs the update.

### “The error travels backward.”
As an intuition, yes. Mathematically, derivatives of the loss with respect to intermediate quantities are propagated backward.

### “Backpropagation is a special kind of calculus.”
It is repeated application of the ordinary chain rule on a computation graph.

### “Automatic differentiation means derivatives are no longer important.”
Understanding derivatives is what lets you understand what automatic differentiation is doing.

---

## 🔬 Failure mode: bad gradient flow

A model can have the right architecture and still train poorly because gradients vanish or explode.

For a long product

$$\prod_i g_i$$

the magnitude can become tiny or huge depending on the factors.

That is why modern deep-learning architecture is partly about making gradient flow practical.

---

## 🧩 Exercises

**Level 1 — Calculate**

1. For $L=(wx-y)^2$, derive $\partial L/\partial w$.
2. For $L=(wx+b-y)^2$, derive both $\partial L/\partial w$ and $\partial L/\partial b$.

**Level 2 — Explain**

3. Explain the chain rule using $w\to z\to h\to\hat y\to L$.
4. Explain the difference between backpropagation and gradient descent.

**Level 3 — Investigate**

5. Build a two-layer network in NumPy and estimate gradients with finite differences.
6. Compare the numerical gradients with PyTorch autograd.

**Level 4 — Research bridge**

7. Investigate vanishing gradients in sigmoid networks.
8. Study how residual connections affect gradient flow.

---

## 🏁 Mastery gate

Move forward only if you can:

- derive a simple gradient by hand;
- explain every factor in a chain-rule product;
- compute $\partial L/\partial w$ and $\partial L/\partial b$ for a linear neuron;
- backpropagate through a ReLU hidden unit;
- distinguish gradient computation from parameter update;
- explain vanishing and exploding gradients;
- verify a hand calculation using automatic differentiation.

---

## What you should remember

> **Backpropagation is the chain rule applied backward through a computation graph.**

The essential equation is

$$
\boxed{
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial\text{output}}
\frac{\partial\text{output}}{\partial\text{hidden}}
\frac{\partial\text{hidden}}{\partial\text{pre-activation}}
\frac{\partial\text{pre-activation}}{\partial w}
}
$$

Then an optimizer uses that information:

$$\boxed{\theta\leftarrow\theta-\eta\nabla_\theta L}.$$

Forward pass tells us **what happened**.

Loss tells us **how wrong** we are.

Backpropagation tells us **how each parameter affects the error**.

The optimizer then changes those parameters.

That loop is the mathematical engine of neural-network training.

**Next:** we build the smallest complete neural network and watch every piece work together.
