# Lecture 07 — Partial Derivatives, Gradients and the Chain Rule

> **The Big Question:** If a loss depends on many knobs at once, how can we know what each knob should do next?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2007%20-%20Partial%20Derivatives%2C%20Gradients%20and%20the%20Chain%20Rule/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Chapter 06 showed that a derivative tells us how a loss changes when one parameter changes.

**Today:** A real model has many parameters. We learn partial derivatives, assemble them into a gradient, and use the chain rule to follow influence through several calculations.

**Next:** The gradient tells us the downhill direction. Chapter 08 asks the remaining question: **how far should we move?**

---

## 1. The Problem: One Loss, Two Knobs

Our house model now has two parameters:

$$
\hat y = wx+b
$$

For one three-room house, suppose $x=3$ and the true price is $7$. The loss is

$$
L(w,b)=(3w+b-7)^2.
$$

The machine has **two knobs**: $w$ and $b$.

In Chapter 06 we learned how to ask, *"What happens if I change one number?"* But now there are two numbers. We need two slope measurements:

$$
\frac{\partial L}{\partial w}
\qquad\text{and}\qquad
\frac{\partial L}{\partial b}.
$$

The symbol $\partial$ is read **"partial"**. It means: change this input while temporarily holding the other inputs fixed.

---

## 2. What Would a Solution Need?

A useful learning signal must:

1. tell us how each parameter affects loss;
2. keep the parameters distinguishable;
3. combine all those directions into one object;
4. work when there are millions of parameters;
5. follow influence through a sequence of operations.

That last requirement is the reason the chain rule will appear later. We first earn the simpler idea: **partial derivatives**.

---

## 3. First Attempt: Change Both Knobs Together

Suppose we change $w$ and $b$ at the same time. The loss changes, so we could divide the change in loss by the total change in parameters.

But which knob caused how much of the change?

We have mixed two effects and lost the information we need.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Just change every parameter at once and use one slope."*
>
> That gives one number for many causes. Training needs to know how each parameter should move separately. The cure is to vary **one parameter at a time** while holding the others fixed.

---

## 4. The Discovery: Partial Derivatives

For

$$
L(w,b)=(3w+b-7)^2,
$$

let

$$
e=3w+b-7.
$$

Then $L=e^2$.

When we differentiate with respect to $w$, treat $b$ as a constant:

$$
\frac{\partial L}{\partial w}
=2e\frac{\partial e}{\partial w}
=2e(3).
$$

So

$$
\boxed{\frac{\partial L}{\partial w}=6(3w+b-7)}.
$$

Likewise, while differentiating with respect to $b$, $w$ is held fixed:

$$
\frac{\partial L}{\partial b}=2e(1).
$$

Therefore

$$
\boxed{\frac{\partial L}{\partial b}=2(3w+b-7)}.
$$

At $w=1$ and $b=0$:

$$
\hat y=3,
\qquad
L=(3-7)^2=16,
$$

and

$$
\frac{\partial L}{\partial w}=6(-4)=-24,
\qquad
\frac{\partial L}{\partial b}=2(-4)=-8.
$$

Both are negative, so increasing either parameter would locally reduce the loss. But $w$ has a much stronger effect here: its slope has magnitude 24 instead of 8.

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Two steering wheels. Turn only one while holding the other still and observe which way the car moves. |
| ✏️ **Numbers** | At $(w,b)=(1,0)$ the slopes are $(-24,-8)$. Both say “move upward,” but the $w$ direction is steeper. |
| 🎓 **Abstraction** | A partial derivative measures local sensitivity to one coordinate while the others are fixed. |

---

## 5. From Many Slopes to One Object

A model might have parameters

$$
\theta_1,\theta_2,\ldots,\theta_n.
$$

We can compute one partial derivative for each:

$$
\frac{\partial L}{\partial\theta_1},
\frac{\partial L}{\partial\theta_2},
\ldots,
\frac{\partial L}{\partial\theta_n}.
$$

But training code needs a single object. Put those numbers into a vector in the same order as the parameters:

$$
\boxed{
\nabla_{\theta}L=
\begin{bmatrix}
\partial L/\partial\theta_1\\
\partial L/\partial\theta_2\\
\vdots\\
\partial L/\partial\theta_n
\end{bmatrix}}
$$

This is the **gradient**.

The gradient is not "the loss." It is not even "the slope" in the one-dimensional sense. It is a vector of all the local slope information at once.

---

## 6. Geometry: Why the Gradient Points Uphill

Imagine the loss as a surface over two parameters, $(w,b)$.

At your current location, there are infinitely many directions you could walk. Each direction produces a directional rate of change.

The gradient picks the direction of **steepest local increase** under ordinary Euclidean geometry.

Therefore

$$
-\nabla L
$$

points in the direction of steepest local decrease.

The word **local** matters. The gradient tells us what the surface is doing *near our current point*. It does not promise a global minimum.

> 🧠 **Think**
>
> If $\nabla L=[-24,-8]^T$, which coordinate direction would reduce loss? Why does the minus sign in gradient descent reverse both entries?

---

## 7. Shape Is a Safety Check

Suppose

$$
\theta\in\mathbb R^3.
$$

Then the gradient must contain one number per parameter:

$$
\nabla_\theta L\in\mathbb R^3.
$$

If the parameter vector has shape `(3,)`, the gradient must have shape `(3,)`.

This sounds obvious. In large tensor programs, it saves hours.

```text
parameters:  (3,)
              ↓ one slope per coordinate
gradient:    (3,)
```

A mismatch is evidence that a formula or an implementation is wrong.

---

## 8. The Problem Again: What If the Loss Has a Chain Inside It?

Now consider a tiny network:

$$
w \longrightarrow z=wx \longrightarrow \hat y=z+b \longrightarrow L=(\hat y-y)^2.
$$

The loss depends on $w$, but only **through** intermediate quantities $z$ and $\hat y$.

We could expand everything and differentiate the giant expression. But a real neural network may contain hundreds of operations.

We need a rule that lets us pass local effects backward through the computation.

---

## 9. First Attempt: Differentiate the Whole Expression at Once

Write

$$
L=(wx+b-y)^2.
$$

This works for a tiny expression. But it hides the structure.

Suppose instead the model had ten intermediate steps. Expanding everything would become a giant algebraic expression, easy to get wrong and hard to reuse.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Always expand the whole network before differentiating."*
>
> Expansion destroys the modular structure of the computation. We want to know the contribution of each small operation so the same machinery works for a huge graph.

This is exactly what the chain rule gives us.

---

## 10. The Discovery: The Chain Rule

If

$$
y=f(g(x)),
$$

then

$$
\boxed{\frac{dy}{dx}=\frac{dy}{dg}\frac{dg}{dx}}.
$$

Read it as:

> **overall local effect = downstream local effect × upstream local effect.**

For our tiny network, define

$$
z=wx,
$$
$$
\hat y=z+b,
$$
$$
L=(\hat y-y)^2.
$$

Then

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
\frac{\partial z}{\partial w}.
$$

Calculate each piece:

$$
\frac{\partial L}{\partial \hat y}=2(\hat y-y),
$$

$$
\frac{\partial \hat y}{\partial z}=1,
$$

$$
\frac{\partial z}{\partial w}=x.
$$

Multiply:

$$
\boxed{\frac{\partial L}{\partial w}=2(\hat y-y)x}.
$$

This is the same result we would get by expanding everything, but now the path is visible.

---

## 11. A Four-Number Hand Calculation

Take

$$
x=3,\quad w=1,\quad b=0,\quad y=7.$$

Forward:

$$
z=wx=3,$$
$$
\hat y=z+b=3,$$
$$
L=(3-7)^2=16.
$$

Backward:

$$
\frac{\partial L}{\partial\hat y}=2(3-7)=-8,
$$
$$
\frac{\partial\hat y}{\partial z}=1,
$$
$$
\frac{\partial z}{\partial w}=3.
$$

Therefore

$$
\frac{\partial L}{\partial w}=(-8)(1)(3)=-24.
$$

Exactly the same slope we found using the partial derivative in §4.

That agreement is important: **two different routes, one mathematical answer.**

---

## 12. The Computational Graph

The chain rule becomes easier to see as a graph:

```mermaid
flowchart LR
    x --> z[ z = w × x ]
    w --> z
    z --> p[ ŷ = z + b ]
    b --> p
    p --> L[ L = (ŷ - y)² ]
    y --> L
```

Forward pass:

$$
(w,x,b)\rightarrow z\rightarrow\hat y\rightarrow L.
$$

Backward pass:

$$
L\rightarrow\hat y\rightarrow z\rightarrow w.
$$

The forward pass computes values. The backward pass computes sensitivities.

That is the conceptual heart of **backpropagation**, which will be developed later in the course.

---

## 13. What About More Than One Path?

Here is a subtle point worth earning now. Suppose a parameter influences the loss through two different paths.

Then both paths contribute to the total derivative.

For example,

$$
z=wx,
\qquad
q=w+z,
\qquad
L=q^2.
$$

There are two ways $w$ affects $q$:

1. directly through $w$;
2. indirectly through $z=wx$.

So

$$
\frac{dq}{dw}=1+x.
$$

The graph forces us to **add contributions from separate paths**. This is why computational graphs are more than drawing aids: they tell us how derivatives must be accumulated.

---

## 14. History Lens — Gottfried Wilhelm Leibniz and the Language of Change

Imagine mathematics in the late seventeenth century. Isaac Newton and Gottfried Wilhelm Leibniz are independently building methods for describing changing quantities.

Leibniz introduced notation such as $dx$, $dy$, and the integral sign that became the language many scientists use today. His notation made relationships between changing quantities visible on the page.

The useful lesson for machine learning is not a story about who "won." It is that notation can turn a difficult calculation into a reusable language. The symbol $\partial$ later lets us say, precisely, **"change this variable while holding the others fixed."** The chain rule then lets those local effects compose.

---

## 15. 🔬 The Experiment

Predict before you run the notebook.

For

$$
L(w,b)=(3w+b-7)^2
$$

start at $(w,b)=(1,0)$.

1. Predict the loss.
2. Predict $\partial L/\partial w$.
3. Predict $\partial L/\partial b$.
4. Predict which parameter has the larger local effect.
5. Change only $b$ by $+0.01$. Does the loss increase or decrease?

Then try the same with only $w$ changed by $+0.01$.

The point is not to guess correctly. The point is to compare a prediction with an experiment and explain the difference.

---

## Distinctions That Matter

| Confusable pair | Difference |
|---|---|
| derivative vs partial derivative | one input vs one coordinate of a multi-input function |
| gradient vs loss | direction/sensitivity information vs a scalar score |
| gradient vs parameter vector | slopes vs the knobs being adjusted |
| chain rule vs gradient descent | derivative calculation vs parameter update |
| local vs global | nearby behavior vs behavior over the whole landscape |

---

## What We Discovered

1. **Many parameters require many local sensitivities.**
2. **A partial derivative isolates one parameter while holding the others fixed.**
3. **The gradient packages every partial derivative into one vector.**
4. **The negative gradient gives the steepest local decrease direction under Euclidean geometry.**
5. **The chain rule composes local effects through a sequence of operations.**
6. **A computational graph makes those dependencies explicit.**

---

## Mathematics We Built

$$
L(w,b)=(3w+b-7)^2
$$

$$
\frac{\partial L}{\partial w}=6(3w+b-7),
\qquad
\frac{\partial L}{\partial b}=2(3w+b-7)
$$

$$
\nabla_\theta L=
\begin{bmatrix}
\partial L/\partial\theta_1\\
\vdots\\
\partial L/\partial\theta_n
\end{bmatrix}
$$

$$
\frac{dy}{dx}=\frac{dy}{dg}\frac{dg}{dx}
$$

$$
\frac{\partial L}{\partial w}=2(\hat y-y)x
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $\partial$ | “partial” | change one variable while fixing the others | `partial` |
| $\nabla$ | “nabla” or “gradient” | vector of partial derivatives | `gradient` |
| $\theta$ | “theta” | model parameters | `theta` |
| $L$ | “loss” | scalar measure of error | `loss` |
| $\eta$ | “eta” | learning rate, used next chapter | `learning_rate` |
| $x$ | “x” | input | `x` |
| $w,b$ | “weight, bias” | trainable parameters in our tiny model | `w`, `b` |

---

## One-Minute Explanation

A derivative tells us how a quantity changes. A partial derivative asks the same question about one coordinate of a multi-variable function. Put all those answers into a vector and you get the gradient. The gradient points uphill, so its negative points downhill. When the loss depends on a parameter through several intermediate calculations, the chain rule lets us multiply the local effects along each path. That is how a large network can turn a final error into information about individual parameters.

---

## Exercises

### Level 1 — Observe

Look at the surface of $L(w,b)=(3w+b-7)^2$ in the notebook. Find a point where both partial derivatives are close to zero.

### Level 2 — Calculate

For $x=2$, $w=4$, $b=1$, and $y=10$, calculate $z$, $\hat y$, $L$, $\partial L/\partial w$, and $\partial L/\partial b$.

### Level 3 — Derive

Derive $\partial L/\partial w$ for

$$
L=(wx+b-y)^2
$$

without first expanding the square.

### Level 4 — Investigate

Change exactly one of $w$, $b$, or the learning rate in the notebook. Predict what will happen to the gradient or loss before running the cell.

### Level 5 — Design

Design a three-parameter toy model whose loss has a known minimum. Explain what the three gradient coordinates should mean before writing any code.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Treating $\partial L/\partial w$ and $\partial L/\partial b$ as the same thing | each holds a different coordinate fixed |
| Calling the gradient a scalar | it contains one value per parameter |
| Thinking $-\nabla L$ is always globally downhill | it is the steepest local decrease direction under the stated geometry |
| Applying the chain rule by adding derivatives | composition requires multiplication along a path |
| Forgetting to add contributions from separate paths | total derivative includes every path through which a parameter acts |
| Ignoring shapes | shape mismatches often reveal a mathematical mistake |

---

## Socratic Questions

1. Why does changing two parameters simultaneously hide information?
2. Why does the gradient need exactly one coordinate per parameter?
3. Why is the gradient an uphill direction rather than a downhill direction?
4. Why do chain-rule factors multiply along a path?
5. Why must contributions from separate paths be added?
6. When might the steepest local direction be a poor long-distance strategy?

---

## 🔭 Bridge to Chapter 08

We now know **which way** to move: opposite the gradient.

But knowing the direction is not enough.

Take a step that is too small and learning can crawl. Take a step that is too large and we can jump across the valley or become unstable.

> **Next: Gradient Descent — how to turn a direction into a controlled sequence of parameter updates.**
