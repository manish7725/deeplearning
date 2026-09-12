# Blog 08 — Derivatives: The Compass for Learning

<!-- NOTEBOOK-LAB-NAV -->

## 🧪 Interactive Lab

The explanation and the hands-on experiment now live together: the notebook contains the complete runnable lab for this lesson.

**[📓 Open the notebook on GitHub](https://github.com/manish7725/deeplearning/blob/main/notebooks/08-derivatives-the-compass.ipynb)**  · **[▶ Open the notebook in Google Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/notebooks/08-derivatives-the-compass.ipynb)**

Run the cells, change the values, observe the result, and then return to this blog to connect the experiment back to the idea.

## 1. Start with a simple curve

Consider

$$
f(x)=x^2
$$

At $x=2$:

$$
f(2)=4
$$

At $x=3$:

$$
f(3)=9
$$

The function rises as $x$ increases.

The derivative tells us **how quickly** it rises or falls at a particular point.

---

## 2. Derivative of a square

For

$$
f(x)=x^2
$$

the derivative is

$$
f'(x)=2x
$$

At $x=3$:

$$
f'(3)=6
$$

So the curve is increasing with slope 6 at that point.

At $x=-3$:

$$
f'(-3)=-6
$$

The negative sign tells us the function decreases as we move to the right.

---

## 3. The derivative is a direction signal

Think of walking on a hill.

- positive slope → uphill
- negative slope → downhill
- zero slope → locally flat

For optimization, this is exactly the information we need.

If we want to minimize something, we want to move **against the slope**.

---

## 4. A numerical approximation

We can estimate a derivative without knowing calculus first.

$$
f'(x)\approx\frac{f(x+h)-f(x)}{h}
$$

For $f(x)=x^2$, $x=2$, and $h=0.001$:

$$
\frac{2.001^2-2^2}{0.001}\approx4.001
$$

The exact derivative is

$$
f'(2)=4
$$

The estimate is close.

---

## 5. Why neural networks care

Suppose a model has one parameter $w$ and its loss is

$$
L(w)=(w-3)^2
$$

Then

$$
\frac{dL}{dw}=2(w-3)
$$

At $w=5$:

$$
\frac{dL}{dw}=4
$$

The positive derivative says increasing $w$ would increase loss locally.

To reduce loss, we should move $w$ downward.

---

## 6. The optimization connection

The update rule will eventually be

$$
w_{new}=w_{old}-\eta\frac{dL}{dw}
$$

where $\eta$ is the learning rate.

Notice the minus sign.

We move in the direction opposite the derivative.

This is the core idea behind gradient descent.

---

## 7. Partial derivatives

A neural network has many parameters:

$$
\theta_1,\theta_2,\ldots,\theta_n
$$

The loss depends on all of them:

$$
L=L(\theta_1,\theta_2,\ldots,\theta_n)
$$

We therefore calculate partial derivatives:

$$
\frac{\partial L}{\partial\theta_1},
\frac{\partial L}{\partial\theta_2},
\ldots,
\frac{\partial L}{\partial\theta_n}
$$

Together they form the **gradient**:

$$
\nabla_\theta L=
\begin{bmatrix}
\frac{\partial L}{\partial\theta_1}\\
\vdots\\
\frac{\partial L}{\partial\theta_n}
\end{bmatrix}
$$

The gradient points toward the direction of steepest local increase.

So $-\nabla L$ points toward steepest local decrease.

---

## 8. The chain rule: the bridge to backpropagation

Suppose

$$
y=f(g(x))
$$

Then

$$
\frac{dy}{dx}=\frac{dy}{dg}\frac{dg}{dx}
$$

This is the **chain rule**.

It says that when functions are composed, their local effects multiply.

Neural networks are compositions of functions, so the chain rule becomes one of their most important mathematical tools.

---

## 9. A tiny neural example

Suppose

$$
z=wx
$$

and

$$
L=(z-y)^2
$$

We want $dL/dw$.

By the chain rule:

$$
\frac{dL}{dw}
=
\frac{dL}{dz}\frac{dz}{dw}
$$

Now

$$
\frac{dL}{dz}=2(z-y)
$$

and

$$
\frac{dz}{dw}=x
$$

Therefore

$$
\boxed{\frac{dL}{dw}=2(z-y)x}
$$

That is already a miniature version of backpropagation.

---

## 10. Code: numerical derivative

```python
def f(x):
    return x ** 2

x = 3.0
h = 1e-5

approx = (f(x + h) - f(x)) / h
print(approx)
```

You should get something very close to 6.

---

## 11. PyTorch can calculate derivatives

```python
import torch

w = torch.tensor(5.0, requires_grad=True)
loss = (w - 3) ** 2

loss.backward()

print(loss.item())
print(w.grad.item())
```

The gradient should be $4$.

PyTorch has automatically performed differentiation for us.

---

## Think Like a Scientist 🧠

For

$$
L(w)=(w-4)^2
$$

find

$$
\frac{dL}{dw}
$$

Then evaluate it at $w=1$, $w=4$, and $w=6$.

Ask:

- At which point is the slope zero?
- Which points have a negative gradient?
- Which direction should we move to reduce loss?

---

## What you should remember

> **A derivative tells us how a quantity changes when an input changes.**

For learning, derivatives tell us how the loss changes when parameters change.

The key bridge is

$$
\text{parameters}
\rightarrow
\text{gradient of loss}
\rightarrow
\text{parameter update}
$$

Next we turn that idea into an algorithm.

> **Next: gradient descent — teaching a model to improve.**

---

# 📚 Go Deeper — Learn Calculus Three Ways

**3Blue1Brown** is the visual route: use it when derivative notation feels abstract and you want to understand slope, local change and geometry. The same visual thinking later becomes useful for gradients and neural networks. citeturn0youtube30turn0youtube31

**MrJensenMath10** is the practice route: strengthen algebra, functions, slopes and exponentials until derivative manipulation becomes comfortable.

**Welch Labs** is the ML route: its neural-network series uses high-school-level calculus to derive backpropagation and connects derivatives directly to training. citeturn0search8

Use **Frame Zero** for first-principles ML intuition and later **ZacharyLLM/Visual Kernel** to see where differentiation and optimization appear in modern models.

### The standard we want

For every derivative in this course, aim to understand all three:

$$
\boxed{
\text{geometric meaning}
\leftrightarrow
\text{symbolic derivation}
\leftrightarrow
\text{numerical verification}
}
$$

If you can do all three, calculus stops being a prerequisite and becomes a working tool.
