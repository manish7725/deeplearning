# Lecture 06 — Derivatives: The Compass for Learning

> **The Big Question:** If we can measure how wrong a model is, how can we discover which direction makes it less wrong?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2006%20-%20Derivatives%3A%20The%20Compass%20for%20Learning/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Chapter 05 introduced the smallest trainable model and the idea that its parameters control its predictions.

**Today:** We discover the derivative as a precise answer to one question: *if I nudge a parameter, what happens to the loss?*

**Next:** Chapter 07 has more than one parameter, so one derivative becomes a gradient and the chain rule becomes necessary.

---

## 1. The Problem: A Model Has a Knob

Start with the smallest useful learning problem:

$$
L(w)=(w-3)^2.
$$

The best parameter is $w=3$, because there $L=0$.

Suppose our model currently uses $w=5$. Then

$$
L(5)=(5-3)^2=4.
$$

The obvious question is:

> **Should we increase $w$ or decrease it? And how strongly?**

The loss value 4 tells us *how bad* the current point is. It does not tell us which nearby move is good.

---

## 2. What Would a Solution Need?

A useful learning signal should:

1. tell us what happens for a tiny change;
2. tell us the direction of improvement;
3. be local enough to calculate from the current point;
4. become exact in the limit of an infinitesimally small change.

Requirement 4 sounds abstract. We can build toward it without assuming calculus.

---

## 3. First Attempt: Try a Large Jump

At $w=5$, compare nearby values:

| $w$ | $L(w)$ |
|---:|---:|
| 4 | 1 |
| 5 | 4 |
| 6 | 9 |

Moving from 5 to 4 helps. Moving from 5 to 6 hurts.

So the direction is clear: **move left**.

But how strong is the signal? The change from 5 to 4 is a whole unit. In a large model, a huge jump can hide what is happening locally.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Just compare the loss at two nearby points and call the difference the slope."*
>
> The raw loss difference depends on how large the step was. A move of 1 and a move of 0.001 cannot be compared directly. We need **change in loss per unit change in input**.

---

## 4. The Discovery: Average Rate of Change

Take a small step $h$.

The change in input is $h$ and the change in output is

$$
f(x+h)-f(x).
$$

So the change **per unit input** is

$$
\frac{f(x+h)-f(x)}{h}.
$$

This is the **secant slope**: the slope of the line joining two nearby points on the curve.

For $f(x)=x^2$, $x=2$, and $h=0.001$:

$$
\frac{2.001^2-2^2}{0.001}
=
\frac{4.004001-4}{0.001}
=4.001.
$$

The exact local slope will be 4. The secant gives 4.001 because the two points are close, not identical.

| Level | The same idea |
|---|---|
| 💡 **Intuition** | How steep is a road? Measure how much altitude changes for each metre you travel. |
| ✏️ **Numbers** | From $(2,4)$ to $(2.001,4.004001)$, the rise is $0.004001$ over a run of $0.001$: slope $4.001$. |
| 🎓 **Abstraction** | Average local change is $[f(x+h)-f(x)]/h$. |

---

## 5. The Discovery: The Derivative

Now shrink the step.

We do not want the slope over a tiny interval. We want the slope **at the point itself**.

Define the derivative as the limit:

$$
\boxed{
 f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}}
$$

Read $h\to0$ as *"make the step smaller and smaller until the average slope settles on the exact local slope."*

For $f(x)=x^2$:

$$
\begin{aligned}
f'(x)
&=\lim_{h\to0}\frac{(x+h)^2-x^2}{h}\\
&=\lim_{h\to0}\frac{x^2+2xh+h^2-x^2}{h}\\
&=\lim_{h\to0}(2x+h)\\
&=2x.
\end{aligned}
$$

The derivative is therefore

$$
\boxed{f'(x)=2x}.
$$

At $x=3$, $f'(3)=6$. At $x=-3$, $f'(-3)=-6$.

The sign tells the direction in which the function is changing.

---

## 6. Why This Solves the Learning Problem

Return to the loss

$$
L(w)=(w-3)^2.
$$

Differentiate:

$$
\boxed{\frac{dL}{dw}=2(w-3)}.
$$

At $w=5$:

$$
\frac{dL}{dw}=4.
$$

A positive derivative means that increasing $w$ increases loss locally. To reduce loss, move **downward in $w$**.

At $w=1$:

$$
\frac{dL}{dw}=-4.
$$

Now increasing $w$ decreases loss. The sign flips exactly where the minimum lies.

At $w=3$:

$$
\frac{dL}{dw}=0.
$$

The loss is locally flat.

> 💡 The derivative has become a **compass**: its sign tells us left or right; its magnitude tells us how steeply the loss is changing.

---

## 7. The Geometry

Plot $L(w)=(w-3)^2$.

The curve is a bowl. At the left side the tangent points downward as we move right, so the derivative is negative. At the right side it points upward, so the derivative is positive. At the bottom the tangent is horizontal.

```text
loss
  ↑
  |       /\
  |      /  \
  |     /    \
  |____/______\____→ w
          3
       minimum
```

The derivative is the slope of the tangent line, not the height of the curve.

That distinction matters:

| Quantity | Question |
|---|---|
| $L(w)$ | How wrong are we? |
| $dL/dw$ | Which way does loss change if $w$ moves? |

---

## 8. What If the Curve Is Not $x^2$?

The derivative idea is not tied to one formula.

For

$$
f(x)=3x+1,
$$

the slope is always 3.

For

$$
f(x)=x^3,
$$

the derivative is

$$
f'(x)=3x^2.
$$

Different functions have different slope rules, but the question is unchanged:

> **How much does the output change for an infinitesimal change in the input?**

---

## 9. The Problem Again: Neural Networks Have Many Parameters

A realistic model does not have just one knob.

It might have

$$
\theta_1,\theta_2,\ldots,\theta_n.
$$

The loss becomes

$$
L=L(\theta_1,\theta_2,\ldots,\theta_n).
$$

We can still ask the same question — but now separately for each parameter.

That is the problem Chapters 07 and 08 will solve.

---

## 10. 🔬 Experiment: Finite Differences

The derivative should agree with a numerical estimate when $h$ is small.

For $f(x)=x^2$ at $x=3$:

$$
\frac{f(3+h)-f(3)}{h}
$$

Try $h=10^{-1},10^{-2},10^{-3},10^{-4},10^{-5}$.

Predict first: **does the estimate approach 6, move away from 6, or oscillate?**

The notebook tests your prediction.

---

## 11. History Lens — Isaac Newton and Gottfried Wilhelm Leibniz

Imagine trying to describe motion in the seventeenth century. Position changes with time, speed changes, and the ordinary algebra of fixed quantities is not enough.

Isaac Newton developed methods for changing quantities while studying mechanics. Gottfried Wilhelm Leibniz independently developed differential notation that later became a standard language for change.

The important lesson for this course is not choosing a winner. It is that calculus made **local change computable**. Machine learning uses exactly that power: the model changes its parameters, and the loss tells us how those changes matter.

---

## 12. Distinctions That Matter

| Confusable pair | Difference |
|---|---|
| value vs derivative | height of a function vs local rate of change |
| average slope vs derivative | finite interval vs limiting local slope |
| derivative vs loss | direction/sensitivity vs error magnitude |
| positive derivative vs good result | positive means uphill in that coordinate, not that the model is good |
| zero derivative vs global minimum | local flatness does not prove the lowest point globally |

---

## 13. What We Discovered

1. **A loss value says how wrong we are, not how to improve.**
2. **Dividing output change by input change gives a local rate of change.**
3. **The derivative is the limiting local slope.**
4. **The derivative's sign gives a direction; its magnitude gives steepness.**
5. **This becomes a learning signal because parameters can be changed in the direction that reduces loss.**

---

## Mathematics We Built

$$
\frac{f(x+h)-f(x)}{h}
$$

$$
 f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
$$

$$
\frac{d}{dx}x^2=2x
$$

$$
\frac{d}{dw}(w-3)^2=2(w-3)
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $f(x)$ | “f of x” | output of a function | `f(x)` |
| $h$ | “h” | small change in input | `h` |
| $f'(x)$ | “f prime of x” | derivative | `derivative` |
| $\lim$ | “limit” | value approached as something gets arbitrarily close | `limit` |
| $L$ | “loss” | scalar error measure | `loss` |
| $w$ | “weight” | one trainable parameter | `w` |

---

## One-Minute Explanation

A derivative answers one question: *if I move the input a tiny amount, how does the output change?* We estimate it by change in output divided by change in input, then take the limit as the step becomes tiny. In machine learning, the output is the loss and the input is a model parameter. The derivative therefore tells us which way that parameter should move to reduce loss.

---

## Exercises

### Level 1 — Observe

Look at the derivative of $L(w)=(w-3)^2$. Identify where it is negative, zero, and positive.

### Level 2 — Calculate

Compute $f'(2)$ for $f(x)=x^2$, and estimate it using $h=0.01$.

### Level 3 — Derive

Derive the derivative of $x^2$ from the limit definition. Do not quote the power rule.

### Level 4 — Investigate

Change only $h$ in the notebook's finite-difference experiment. Explain why the estimate first improves and can eventually become noisy because of floating-point arithmetic.

### Level 5 — Design

Invent a simple loss curve with two local minima. Explain why a derivative alone cannot tell you which minimum is globally best.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Calling the loss itself a slope | loss is a value; derivative is a rate of change |
| Using a raw difference without dividing by $h$ | the result depends on the arbitrary step size |
| Thinking derivative zero means “perfect model” | it only means locally flat |
| Thinking a positive derivative means loss is positive | it means loss increases as the parameter increases locally |
| Using an enormous finite-difference step | it measures average behavior over a wide interval, not local behavior |

---

## Socratic Questions

1. Why must we divide by the size of the step?
2. Why does making $h$ smaller move us toward the tangent slope?
3. Why can a derivative be zero away from a global minimum?
4. Why is the sign more useful for optimization than the loss value alone?
5. What information do we lose when a model has several parameters but we measure only one combined change?

---

## 🔭 Bridge to Chapter 07

A real network has many parameters. One derivative is no longer enough.

> **Next: Partial Derivatives, Gradients and the Chain Rule — one compass reading for every parameter, and a way to pass that information backward through a computation.**
