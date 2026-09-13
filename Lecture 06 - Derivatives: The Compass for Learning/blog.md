# Lecture 06 — Derivatives: The Compass for Learning

> **The Big Question:** What does it mean to measure a rate of change at a single instant, when nothing has had time to change?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2006%20-%20Derivatives%3A%20The%20Compass%20for%20Learning/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We found special directions in a transformation.

Now we have a different problem: during learning, the loss changes whenever we move a parameter.

We need a way to measure **how fast something changes right here**.

> **How can we measure the exact steepness of a curve at one point?**

**Next → Derivatives.**

## 1. The Problem: The Speedometer Paradox

You are in a car. The speedometer reads **60 km/h**.

Here is a question that sounds childish and is not: *what does that number mean right now?*

Speed is distance divided by time. So to measure it you need a distance and a time. Fine — over the last hour the car travelled 60 km, so the speed is 60 km/h. Over the last minute it travelled 1 km. Over the last second, about 16.7 metres.

Now go all the way. What distance does the car cover **at this exact instant**?

Zero. An instant has no duration, so nothing moves in it. And how much time passes in an instant? Also zero. So the speed at this instant is

$$
\frac{0}{0}
$$

which is not 60. It is not anything at all — $0/0$ is undefined, and dividing by zero is the one thing arithmetic absolutely refuses to do.

And yet the needle is sitting at 60, perfectly steady. The car really does have a speed right now. A photograph taken at this instant would catch the needle at 60.

> 🧠 **Think** — Something is wrong with our definition, not with the car. We have a quantity that obviously exists in the world and a formula that produces nonsense when we ask for it at a point. That gap is where an entire branch of mathematics comes from.

### Why this course cares

This is not a detour into physics. It is a debt coming due.

In Chapter 1 we asked *"which way should I nudge $w$ to make the loss smaller?"* and answered it by computing the loss at $w$ and at a nearby $w+h$, taking the difference, dividing by $h$, and then — this is the part we glossed over — **letting $h$ shrink to nothing** and seeing what survived. We got $15(w-2)$, called it a compass, and moved on.

That manoeuvre is exactly the speedometer paradox. We divided by a gap and then made the gap vanish. If that is illegitimate, then Chapter 1's compass is illegitimate, and so is every gradient in the rest of this course.

So: either we make this rigorous, or we admit the whole subject rests on hand-waving.

---

## 2. What Would a Solution Need?

Before inventing anything, let us be precise about what a good answer must do.

1. **Produce a single number at a single point.** "The speed between 2 and 3 seconds" is not what the needle shows. We want the speed *at* 2 seconds.
2. **Never actually divide by zero.** Whatever we do, the arithmetic must stay legal at every step.
3. **Agree with common sense where common sense works.** For a car moving at a steady 60, the answer at every instant must be 60.
4. **Tell us direction as well as size.** Chapter 1 needed to know whether the loss was rising or falling, not merely how fast.
5. **Be computable.** If we cannot evaluate it, it cannot steer a training loop.

Requirement 2 is the hard one, and the whole chapter is about a trick for satisfying it.

---

## 3. First Attempt: Average Speed

Let us make the car concrete so we can compute. Suppose its distance travelled, in metres, after $t$ seconds is

$$
s(t) = t^2
$$

So after 1 second it has gone 1 m, after 2 seconds 4 m, after 3 seconds 9 m. It is speeding up — each second covers more ground than the last.

**Average speed** over a stretch of time is honest arithmetic: distance covered divided by time taken. From $t = 2$ to $t = 3$:

$$
\frac{s(3) - s(2)}{3 - 2} = \frac{9 - 4}{1} = 5 \text{ m/s}
$$

This works. It satisfies requirements 2, 3 and 5. But it does **not** satisfy requirement 1, and that failure is not a technicality — the number 5 does not describe the car at $t=2$ at all. At $t=2$ the car is going slower than 5; by $t=3$ it is going faster. Five is a summary of a whole second's worth of behaviour, and the car's speed changed throughout.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Just use a really short interval and call it the instantaneous speed."*
>
> This is closer to right than it looks, and it is still wrong as stated. How short is short enough? A tenth of a second? A microsecond? Every answer you give is still an **average** over some stretch, and there is always a shorter stretch giving a different answer. "Short enough" is not a definition — it is a decision you are quietly making for the reader.
>
> The fix is not to pick a small gap. It is to stop picking, and instead study **what happens to the answer as the gap shrinks**.

---

## 4. The Discovery: Shrink the Gap and Watch the Pattern

Here is the move that resolves everything, and it is a change of question rather than a change of arithmetic.

Do not ask *"what is the answer when $h = 0$?"* — that question is illegal. Ask instead:

> **As $h$ gets smaller and smaller, what number is the answer heading towards?**

That is a legal question, because at every step $h$ is a genuine non-zero number and the division is genuine division. We never divide by zero; we simply watch a trend.

Compute the average speed from $t = 2$ to $t = 2+h$ for a sequence of shrinking $h$:

| $h$ (seconds) | from $t=2$ to | average speed |
|---:|---:|---:|
| $1$ | $3$ | $5$ |
| $0.5$ | $2.5$ | $4.5$ |
| $0.1$ | $2.1$ | $4.1$ |
| $0.01$ | $2.01$ | $4.01$ |
| $0.001$ | $2.001$ | $4.001$ |

Look at the right-hand column. The numbers are not wandering. They are marching, unmistakably, towards **4**.

And we can see exactly why, with algebra rather than a table. The average speed over $[2, 2+h]$ is

$$
\frac{s(2+h) - s(2)}{h} = \frac{(2+h)^2 - 2^2}{h}
$$

Expand the square — $(2+h)^2 = 4 + 4h + h^2$:

$$
= \frac{4 + 4h + h^2 - 4}{h} = \frac{4h + h^2}{h}
$$

Now, **while $h$ is still not zero**, we are allowed to cancel it:

$$
= \frac{h(4 + h)}{h} = 4 + h
$$

There it is. The average speed over any interval of length $h$ starting at $t=2$ is exactly $4 + h$. That is why the table reads 5, 4.5, 4.1, 4.01 — it is $4+h$ every time.

And now the question that was illegal becomes easy. We are not asking for $h = 0$. We are asking what $4 + h$ heads towards as $h$ shrinks. The $h$ term withers away and **4** is left standing.

$$
\boxed{\;\text{the speed at } t = 2 \text{ is } 4 \text{ m/s}\;}
$$

> 💡 **Intuition** — The cancellation is the whole trick. Before cancelling, setting $h=0$ gives $0/0$ — nonsense. After cancelling, setting $h \to 0$ gives $4$ — perfectly sensible. Same expression, and the algebra did the work that arithmetic could not.

Notice what we never did: we never divided by zero. At every line, $h$ was a real, non-zero number. We only asked about the *destination* of a journey, not about arriving.

The notation for "what this heads towards as $h$ shrinks to nothing" is:

$$
\lim_{h \to 0} (4 + h) = 4
$$

Read $\lim_{h\to 0}$ aloud as: *"the value this expression closes in on, as $h$ gets arbitrarily close to zero without ever being zero."* The arrow means "approaches", never "equals".

---

## 5. 📜 History Lens — Newton, Leibniz, and a Bishop's Objection

This argument took humanity a very long time, and it was ferociously contested.

In the 1660s–1680s two people arrived at it independently. **Isaac Newton**, in England, was trying to describe motion and gravitation; he needed to talk about a planet's velocity *at a moment*, and called his rates of change *fluxions*. **Gottfried Wilhelm Leibniz**, in Germany, came at it from the geometry of curves and tangent lines, and invented the notation $\frac{dy}{dx}$ that we still use — his notation was better, and it won.

Their followers then spent decades in an ugly priority dispute over who was first. The mathematics survived the argument; both had it.

But the sharpest attack came from outside mathematics. In 1734, **Bishop George Berkeley** published *The Analyst*, pointing out that the method appeared to cheat. You divide by $h$ — which requires $h \ne 0$ — and then you discard $h$ as though it were zero. Which is it? In his famous phrase, these vanishing quantities were:

> "the ghosts of departed quantities"

**Berkeley was right that the reasoning was not yet rigorous.** Calculus worked spectacularly — it predicted planets — but nobody could explain *why* the trick was legitimate. That is an uncomfortable position for a subject to be in, and it lasted roughly 150 years.

The repair came in the 1800s, from **Augustin-Louis Cauchy** and later **Karl Weierstrass**, who defined the limit precisely: not as "$h$ becomes zero", but as "the answer can be forced as close to $L$ as you like by making $h$ small enough." No ghosts, no vanishing — just a statement about how close you can get.

> 🧠 That is exactly the move in §4. We never claimed $h$ becomes zero. We described where the answers were heading. The reason we can write $\lim$ with a straight face is that Cauchy and Weierstrass paid for it.

---

## 6. The Derivative, Defined

Now generalize. Replace the car with any function $f$, and the point $t=2$ with any point $x$:

$$
\boxed{\;f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}\;}
$$

This is the **derivative** of $f$ at $x$ — the instantaneous rate of change. Let us translate every part into English, because the symbols are doing real work:

| Piece | Read it as |
|---|---|
| $f(x+h) - f(x)$ | "how much the output changed" |
| $h$ | "how much the input changed" |
| $\dfrac{f(x+h)-f(x)}{h}$ | "output change per unit of input change" — the average rate over the gap |
| $\lim_{h\to 0}$ | "where that ratio heads as the gap closes" |
| $f'(x)$ | "the rate of change at the single point $x$" |

The three levels, on the same idea:

| Level | The derivative is… |
|---|---|
| 💡 **Intuition** | How fast the output moves when you nudge the input. The speedometer of a function. |
| ✏️ **Numbers** | For $s(t)=t^2$ at $t=2$: nudge the input by $h$ and the output moves by $4h + h^2$, a rate of $4+h$, heading to **4**. |
| 🎓 **Abstraction** | $f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$, when that limit exists. |

### Two notations, one idea

You will meet both, often on the same page:

$$
f'(x) \qquad\text{(Lagrange)} \qquad\qquad \frac{dy}{dx} \qquad\text{(Leibniz)}
$$

$f'(x)$ — read *"f prime of x"* — is compact and good for stating rules.

$\frac{dy}{dx}$ — read *"dee y by dee x"* — is Leibniz's, and it deliberately looks like a fraction of "a tiny change in $y$" over "a tiny change in $x$". It is **not** a fraction; $dy$ and $dx$ are not numbers you can separate. But it is built to *remind* you of $\frac{\Delta y}{\Delta x}$, and that resemblance will make the chain rule in Chapter 7 look almost obvious.

> ⚠️ **Do not** read $\frac{dy}{dx}$ as $d$ times $y$ divided by $d$ times $x$. There is no quantity $d$. The whole symbol $\frac{d}{dx}$ means "the derivative with respect to $x$ of whatever follows."

---

## 7. Deriving the Rules — Not Memorizing Them

Textbooks hand you a table of derivative rules. We are going to *derive* them, because every one falls straight out of the definition, and because a rule you derived once is a rule you can rebuild when you forget it.

### The square, in full

We did $x^2$ at the specific point $x=2$. Do it at a general $x$:

$$
\frac{(x+h)^2 - x^2}{h}
$$

Expand $(x+h)^2 = x^2 + 2xh + h^2$:

$$
= \frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{2xh + h^2}{h}
$$

The $x^2$ terms cancel — they always do, because we subtracted $f(x)$ from $f(x+h)$. Factor out $h$ and cancel it while it is still non-zero:

$$
= \frac{h(2x + h)}{h} = 2x + h
$$

Let $h \to 0$:

$$
\frac{d}{dx}\left(x^2\right) = 2x
$$

Sanity check against §4: at $x = 2$ this gives $2(2) = 4$ ✓, matching the table exactly.

### The cube, to see the pattern

$(x+h)^3 = x^3 + 3x^2h + 3xh^2 + h^3$, so:

$$
\frac{(x+h)^3 - x^3}{h} = \frac{3x^2h + 3xh^2 + h^3}{h} = 3x^2 + 3xh + h^2
$$

Every surviving term except the first still contains an $h$, so all of them wither:

$$
\frac{d}{dx}\left(x^3\right) = 3x^2
$$

### The pattern, stated

$$
\frac{d}{dx}\left(x^2\right) = 2x^1
\qquad
\frac{d}{dx}\left(x^3\right) = 3x^2
\qquad\Longrightarrow\qquad
\boxed{\;\frac{d}{dx}\left(x^n\right) = n\,x^{n-1}\;}
$$

**Bring the exponent down in front, then reduce the exponent by one.** The reason it always works is visible in both expansions above: when you expand $(x+h)^n$, exactly one term has a single $h$ in it — and there are $n$ ways to pick which factor contributes that $h$, which is where the $n$ out front comes from. Every other term has $h^2$ or higher and dies.

### Three more, each in one line

**A constant.** $f(x) = c$ never changes, so $f(x+h) - f(x) = 0$ and the ratio is $0/h = 0$:

$$
\frac{d}{dx}(c) = 0
$$

*A flat road has zero slope.*

**A constant multiple.** If $g(x) = c\,f(x)$, then every difference is multiplied by $c$, and $c$ factors straight out of the limit:

$$
\frac{d}{dx}\big(c\,f(x)\big) = c\,f'(x)
$$

*Doubling a journey's distances doubles its speeds.*

**A sum.** Differences of a sum are the sum of the differences:

$$
\frac{d}{dx}\big(f(x) + g(x)\big) = f'(x) + g'(x)
$$

*Two effects that add, contribute rates that add.*

These four rules — power, constant, constant multiple, sum — are enough for everything in this chapter, and enough to redo Chapter 1 honestly. The product and chain rules arrive in Chapter 7, where we will actually need them.

---

## 8. Back to Chapter 1's Compass

Now we pay the debt.

Chapter 1 studied the loss of our house model with the bias pinned, and derived in closed form:

$$
L(w) = 7.5\,(w-2)^2
$$

Then it asked which way was downhill, expanded $\frac{L(w+h)-L(w)}{h}$, got $15(w-2) + 7.5h$, and discarded the $7.5h$ with a wave of the hand. Let us do it properly — and this time we have machinery, so it takes one line.

First expand the square so the rules apply:

$$
L(w) = 7.5(w-2)^2 = 7.5\left(w^2 - 4w + 4\right) = 7.5w^2 - 30w + 30
$$

Now differentiate term by term, using the sum rule to split it and the power and constant rules on each piece:

$$
\frac{d}{dw}\left(7.5w^2\right) = 7.5 \cdot 2w = 15w
\qquad
\frac{d}{dw}\left(-30w\right) = -30
\qquad
\frac{d}{dw}\left(30\right) = 0
$$

Add them:

$$
L'(w) = 15w - 30 = 15(w - 2)
$$

**Exactly what Chapter 1 claimed**, now with every step legal and no ghosts. The compass that trained our first model is real.

> 🧪 The notebook confirms this numerically in Step 5, and confirms it against Chapter 1's own hand-computed table.

---

## 9. The Geometry: A Secant Becomes a Tangent

There is a picture underneath all of this, and it is worth carrying.

Take the curve $y = f(x)$. Pick your point $\big(x, f(x)\big)$ and a second point a distance $h$ away, $\big(x+h, f(x+h)\big)$. Draw the straight line through both. That line is called a **secant**, and its slope is

$$
\frac{\text{rise}}{\text{run}} = \frac{f(x+h) - f(x)}{h}
$$

— precisely the quantity inside our limit. So **the average rate of change is the slope of a secant line.**

Now slide the second point towards the first. The secant pivots, and as $h$ shrinks it settles into the unique line that grazes the curve at that single point without cutting through it. That limiting line is the **tangent**.

```text
  y                        secant (h large)
  │                    ⟋
  │      curve      ⟋
  │        ___⟋‾‾
  │     ⟋‾‾   ●  ←── tangent (h → 0)
  │  ⟋‾    ⟋‾
  └──────────────────────── x
           x   x+h
```

$$
\textbf{The derivative at a point is the slope of the tangent line there.}
$$

This is why Chapter 1 could talk about "the steepness of the valley wall" and why it made sense to walk downhill. The tangent is the flat board you would balance on the curve at that point, and its tilt is $f'(x)$.

---

## 10. Reading the Sign and the Size

Requirement 4 from §2 asked for direction, not just magnitude. The derivative delivers both, and they are read separately.

**The sign says which way the function is going.**

| $f'(x)$ | The function is | To go downhill you should |
|---|---|---|
| negative | falling as $x$ increases | **increase** $x$ |
| zero | momentarily flat | stop — you are at a peak, valley, or plateau |
| positive | rising as $x$ increases | **decrease** $x$ |

**The size says how steep it is.** $f'(x) = 100$ is a cliff; $f'(x) = 0.01$ is nearly flat.

Check it on our loss, $L'(w) = 15(w-2)$:

| $w$ | $L'(w)$ | Meaning |
|---:|---:|---|
| $0$ | $-30$ | steeply downhill to the right → increase $w$ |
| $1$ | $-15$ | still downhill, half as steep → increase $w$ |
| $2$ | $0$ | flat — the bottom |
| $3$ | $+15$ | uphill to the right → decrease $w$ |
| $4$ | $+30$ | steeply uphill → decrease $w$ |

And this is the origin of the minus sign in the update rule that runs the entire field:

$$
w \leftarrow w - \eta\,L'(w)
$$

When $L'$ is negative, subtracting it *increases* $w$ — which the table says is correct. When $L'$ is positive, subtracting it decreases $w$ — also correct. **One minus sign handles both cases**, with no if-statement anywhere, which is why the rule is written the way it is.

---

## 11. 🔬 The Experiment: Computing Derivatives Numerically

We can now derive derivatives by hand. But a computer often cannot do algebra — it can only evaluate functions. So it estimates the derivative by using a small $h$ and *not* taking the limit:

$$
f'(x) \approx \frac{f(x+h) - f(x)}{h}
$$

> 🧠 **Predict before reading on.** We know $\frac{d}{dx}x^2 = 2x$, so at $x=2$ the true answer is exactly $4$. If we compute the approximation above with smaller and smaller $h$ — $10^{-1}$, then $10^{-3}$, down to $10^{-14}$ — does the error keep shrinking all the way? Should we just use the smallest $h$ we can?

The intuition says smaller $h$ is always better; §4's table marched happily towards 4. Here is what a computer actually produces:

| $h$ | approximation | error |
|---|---:|---:|
| $10^{-1}$ | $4.1$ | $1 \times 10^{-1}$ |
| $10^{-3}$ | $4.001$ | $1 \times 10^{-3}$ |
| $10^{-6}$ | $4.000001$ | $1 \times 10^{-6}$ |
| $10^{-8}$ | $3.99999998$ | $\mathbf{2 \times 10^{-8}}$ ← best |
| $10^{-10}$ | $4.00000033$ | $3 \times 10^{-7}$ |
| $10^{-12}$ | $4.000356$ | $4 \times 10^{-4}$ |
| $10^{-14}$ | $4.0856$ | $9 \times 10^{-2}$ |

The error falls, bottoms out around $h \approx 10^{-8}$, and then **gets worse again** — until at $10^{-14}$ it is as bad as it was at $h = 10^{-1}$.

### Why it turns around

Two errors fight each other, and they pull in opposite directions.

**Truncation error** is the mathematics. We showed the approximation equals $2x + h$ exactly, so it overshoots the true $2x$ by about $h$. This error *shrinks* as $h$ shrinks. That is the left half of the table.

**Rounding error** is the machine. A computer stores numbers with about 16 significant digits. When $h$ is tiny, $f(x+h)$ and $f(x)$ agree in almost every digit — at $h = 10^{-14}$, $f(2+h)$ and $f(2)$ are both $4.000000000000000\ldots$, differing only in the last couple of stored digits. Subtracting them throws away all the agreeing digits and keeps only the noise. Then we **divide that noise by a tiny number**, which magnifies it enormously.

$$
\text{total error} \;\approx\; \underbrace{h}_{\text{mathematics}} \;+\; \underbrace{\frac{\varepsilon}{h}}_{\text{machine}}
$$

One term grows with $h$, the other grows as $h$ shrinks. The sum is smallest where they balance, which for standard double precision is around $h \approx \sqrt{\varepsilon} \approx 10^{-8}$ — exactly where the table bottoms out.

> 💡 **This is the first time in the course that the mathematics and the machine disagree**, and it will not be the last. Calculus says "smaller $h$ is always better." Floating-point arithmetic says "not below $10^{-8}$." Both are right about their own domain, and you have to know both.

> ⚠️ **Practical rule** — when checking a derivative numerically, use $h \approx 10^{-6}$. It is small enough that the truncation error is negligible for a sanity check, and large enough to stay far from the cancellation cliff.

This is also the honest reason deep learning does **not** compute gradients numerically. Besides being inaccurate, estimating the derivative with respect to each of a billion parameters would need a billion extra function evaluations. Chapters 27 and 31 show how to get all of them exactly, in one backward pass, for roughly the cost of one forward pass.

---

## 12. The Second Derivative: Curvature, and Chapter 1's Speed Limit

$L'(w)$ is itself a function of $w$. Nothing stops us differentiating it again:

$$
L'(w) = 15w - 30
\qquad\Longrightarrow\qquad
L''(w) = 15
$$

The **second derivative** is the rate at which the slope itself changes — the **curvature**. Where the first derivative says "how steep", the second says "how sharply the steepness is changing."

| | Meaning | Shape |
|---|---|---|
| $f'' > 0$ | slope increasing | curving upward, like a valley |
| $f'' < 0$ | slope decreasing | curving downward, like a hill |
| $f'' = 0$ | slope constant | a straight line |

Our loss has $L'' = 15$ everywhere — constant positive curvature, which is exactly what a parabola is.

### And now a debt from Chapter 1 gets paid

Chapter 1 §14 ran an experiment: it tried learning rates and found training exploded somewhere around $\eta \approx 0.12$–$0.13$. It stated a formula, $\eta < 2/\lambda_{\max}$, without being able to say what $\lambda_{\max}$ was.

It is the curvature. For a one-parameter loss, the stability condition is

$$
\eta < \frac{2}{L''}
$$

For our loss that is

$$
\eta < \frac{2}{15} \approx 0.1333
$$

Test it. Running gradient descent on $L$ with the bias pinned:

| $\eta$ | after 200 steps |
|---:|---|
| $0.10$ | $w = 2.000000$ ✓ |
| $0.13$ | $w = 1.99993$ ✓ |
| $0.1333$ | $w = 0.19$ — barely hanging on, right at the edge |
| $0.134$ | $w = -12.6$ — drifting away |
| $0.15$ | overflow ✗ |

The predicted threshold is $0.1333$, and the behaviour changes between $0.13$ and $0.134$. **The mathematics said where the machine would break before the machine broke.**

Here is the intuition for *why* curvature sets a speed limit. A gradient step assumes the slope stays roughly constant over the distance you travel. The curvature measures how wrong that assumption goes. In a sharply curved valley ($L''$ large), the slope changes fast, so a long step lands somewhere the slope is completely different — you overshoot, and the overshoot compounds. In a gently curved valley you can stride confidently.

> 🔭 Hold this thought. In Chapter 7 the loss depends on several parameters, and the curvature stops being one number and becomes a matrix — whose largest eigenvalue is the $\lambda_{\max}$ Chapter 1 named. Chapter 5's eigenvalues and this chapter's second derivative are about to meet.

---

## 13. When There Is No Derivative

Requirement 3 said our method should fail honestly when there is no sensible answer. It does.

### A sharp corner

Consider $f(x) = |x|$, the absolute value — a V shape with its point at the origin. What is the slope at $x = 0$?

Approach from the right, with $h > 0$:

$$
\frac{|0+h| - |0|}{h} = \frac{h}{h} = +1
$$

Approach from the left, with $h < 0$:

$$
\frac{|0+h| - |0|}{h} = \frac{-h}{h} = -1
$$

The two sides disagree — $+1$ and $-1$, no matter how small $h$ gets. The limit requires a *single* destination, and there isn't one. So $|x|$ has **no derivative at 0**, and that is the correct answer, not a failure of technique. Everywhere else it is perfectly differentiable: slope $-1$ to the left, $+1$ to the right.

> 🎯 This is not an exotic edge case. Chapter 1 §8 chose squared error partly because $|e|$ has a kink at zero and $e^2$ does not — now you can see precisely what the kink costs. And **ReLU**, the most widely used activation function in deep learning, is exactly this shape: $\max(0, x)$, with a corner at the origin. Chapter 25 explains what frameworks do about it (they pick a value and move on), and why in practice it barely matters.

### A jump

If a function teleports — say, a price that is ₹10 below 1000 sq ft and ₹20 above — then at the jump point the numerator $f(x+h) - f(x)$ stays stubbornly large while $h$ shrinks to nothing. The ratio blows up. **No derivative where a function jumps**, and more basically, a function must be continuous at a point to be differentiable there.

The converse is false: $|x|$ is perfectly continuous at 0 and still has no derivative. Continuity is necessary, not sufficient.

---

## 14. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **$h$ too small numerically** | error grows as $h$ shrinks | Catastrophic cancellation, then division by a tiny number. §11 |
| **$h$ too large** | consistently wrong by about $h$ | Truncation — you measured a secant, not a tangent. §11 |
| **Kink in the function** | left and right disagree forever | No single limit exists. §13 |
| **Confusing $f'$ with $f$** | "the loss is negative" | $L' < 0$ means the loss is *falling*; $L$ itself may be huge. |
| **Reading $\frac{dy}{dx}$ as a fraction** | invalid cancellations | It is one symbol, deliberately shaped like a fraction. §6 |
| **Zero derivative means minimum** | training stops at the wrong place | $f' = 0$ at peaks and plateaus too. §10 |

---

## 15. 🎯 Machine Learning Connection

Every single thing in the rest of this course runs on this chapter.

| This chapter | In machine learning |
|---|---|
| $f'(x)$ — rate of change | how sensitive the loss is to one parameter |
| the sign of $f'$ | which way to nudge that parameter |
| the size of $f'$ | how strongly to nudge it |
| $f' = 0$ | a candidate resting point for training |
| $f''$ — curvature | the speed limit on the learning rate (§12) |
| kinks | ReLU, and why we chose squared error (Ch 1 §8) |
| numerical derivatives | gradient checking — how you verify a hand-derived gradient |

And the update rule that Chapter 1 introduced on faith now reads as plain English:

$$
w \leftarrow w - \eta\,\frac{dL}{dw}
$$

*"Measure how the loss responds to this parameter, then move the parameter the opposite way, by an amount proportional to that response."*

There is no magic left in this line. There is only §6.

---

## 16. Distinctions That Matter

| | |
|---|---|
| **Average rate** — over an interval, a secant slope | **Instantaneous rate** — at a point, a tangent slope |
| $\Delta x$ — a real, measurable gap | $dx$ — notation inside $\frac{dy}{dx}$, not a number |
| $h \to 0$ — approaches | $h = 0$ — illegal, gives $0/0$ |
| $f(x)$ — the value | $f'(x)$ — how fast the value is changing |
| $f'$ — steepness | $f''$ — how fast the steepness changes |
| **Analytic derivative** — exact, from algebra | **Numerical derivative** — approximate, from evaluations |
| **Continuous** — no jumps | **Differentiable** — no jumps *and* no corners |

---

## 17. What We Discovered

1. Speed at an instant appears to be $0/0$, which is nonsense — yet the quantity plainly exists, so the definition was at fault.
2. Average rates over an interval are legal but describe a stretch, not a point.
3. Picking a "small enough" interval is not a definition; it hides a decision.
4. The fix is to change the question: ask where the answer *heads* as the gap closes, never what it *is* when the gap is zero.
5. Algebra does the work arithmetic cannot — cancel $h$ while it is still non-zero, then let it wither.
6. That is the derivative, $f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$, and it is the slope of the tangent line.
7. Every differentiation rule follows from that definition; none needs memorizing.
8. Chapter 1's compass $15(w-2)$ is now derived rather than asserted.
9. The sign of the derivative is why the update rule has a minus sign and needs no if-statement.
10. Numerically, smaller $h$ is **not** always better — truncation and rounding fight, and the winner flips around $h \approx 10^{-8}$.
11. The second derivative is curvature, and it is exactly the quantity that set Chapter 1's learning-rate limit at $2/15$.
12. Corners and jumps genuinely have no derivative, which matters for ReLU and for our choice of squared error.

## 18. Mathematics We Built

$$
\text{average rate} = \frac{f(x+h)-f(x)}{h}
\qquad
f'(x) = \lim_{h \to 0}\frac{f(x+h)-f(x)}{h}
$$

$$
\frac{d}{dx}\left(x^n\right) = n x^{n-1}
\qquad
\frac{d}{dx}(c) = 0
\qquad
\frac{d}{dx}\big(cf\big) = c f'
\qquad
\frac{d}{dx}(f+g) = f' + g'
$$

$$
L(w) = 7.5(w-2)^2
\;\Longrightarrow\;
L'(w) = 15(w-2)
\;\Longrightarrow\;
L''(w) = 15
$$

$$
\text{total numerical error} \approx h + \frac{\varepsilon}{h}
\qquad
\eta < \frac{2}{L''}
$$

## 19. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $h$ | the small gap in the input | `h` |
| $\Delta y$ | the resulting change in output | `f(x+h) - f(x)` |
| $\lim_{h\to 0}$ | "where this heads as $h$ shrinks" | take `h` small, carefully |
| $f'(x)$ | the derivative — rate of change at $x$ | `df(x)` |
| $\frac{dy}{dx}$ | the same thing, Leibniz's way | — |
| $\frac{d}{dx}$ | "the derivative, with respect to $x$, of…" | — |
| $f''(x)$ | the second derivative — curvature | `d2f(x)` |
| $\varepsilon$ | machine precision, about $10^{-16}$ | `np.finfo(float).eps` |
| $\eta$ | learning rate | `learning_rate` |

## 20. One-Minute Explanation

Explain this with no equations at all:

> A car covers zero distance in zero time at any given instant. So how can the speedometer possibly show a number — and how did we get a straight answer out of $0/0$ without cheating?

If your explanation contains the word "limit" without explaining what that word means, you are quoting, not explaining.

---

## 21. Exercises

**Level 1 — Observe.** Look at the table in §10. Between $w=0$ and $w=2$ the derivative goes from $-30$ to $0$ but never turns positive. What does that say about the shape of the loss on that stretch? Now look at the §11 table: at which $h$ is the answer best, and why is it *not* the smallest $h$ in the list? Finally: if a function's derivative is zero at some point, name three different shapes that point could be.

**Level 2 — Calculate (by hand, no code).** For $f(x) = x^2 - 6x + 5$: (a) find $f'(x)$ using the rules from §7; (b) find the $x$ where $f'(x) = 0$; (c) compute $f''(x)$ and use its sign to say whether that point is a minimum or a maximum; (d) verify part (a) from the definition, by expanding $\frac{f(x+h)-f(x)}{h}$ and letting $h \to 0$. Your two answers to (a) and (d) must agree exactly.

**Level 3 — Derive.** Prove the power rule for $n=4$ by expanding $(x+h)^4$ in full and cancelling. Then explain, in one sentence, where the $n$ in $nx^{n-1}$ comes from — what is it counting? Finally, derive $\frac{d}{dx}\left(\frac{1}{x}\right)$ from the definition. (Hint: combine $\frac{1}{x+h} - \frac{1}{x}$ over a common denominator before dividing by $h$.) Does your answer match the power rule with $n = -1$?

**Level 4 — Investigate** (notebook Steps 8–11). Reproduce the U-shaped error curve in §11 for $f(x) = x^2$ at $x=2$, and find the $h$ that minimizes the error. Then repeat at $x = 1000$ instead of $x = 2$. The best $h$ moves — explain why using the fact that rounding error is proportional to the *size of the numbers being subtracted*, not to $h$ alone.

**Level 5 — Design.** The forward difference $\frac{f(x+h)-f(x)}{h}$ has error of about $h$. Design a better estimator using $f(x+h)$ and $f(x-h)$, and work out its error by expanding both terms. (Hint: write out $\frac{f(x+h) - f(x-h)}{2h}$ for $f(x)=x^2$ and see how much error is left.) Why is your version more accurate, what does it cost, and is there a situation in which you could not use it?

---

## 22. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "The derivative is the change in $y$." | It is the change in $y$ **per unit change in $x$** — a rate, not an amount. |
| "$h$ becomes zero." | It never does. It approaches zero; the limit describes the destination. §4 |
| "$\frac{dy}{dx}$ is $dy$ divided by $dx$." | One symbol, shaped suggestively. There is no number $d$. §6 |
| "Smaller $h$ always gives a better numerical answer." | False below $h \approx 10^{-8}$ — rounding error takes over. §11 |
| "$f'(x)=0$ means we found the minimum." | Peaks and plateaus also have zero slope. §10 |
| "Continuous means differentiable." | $\lvert x\rvert$ is continuous at 0 and has no derivative there. §13 |
| "Calculus is a separate topic from machine learning." | The update rule *is* §6 plus a minus sign. §15 |

## 23. Socratic Questions

1. We cancelled $h$ from numerator and denominator "while it is still non-zero", then let it go to zero. Berkeley called this cheating. What exactly makes it legitimate — what is the difference between the two moves?
2. The derivative of a constant is zero. What does that say about a model whose loss does not depend on a parameter at all — and what will gradient descent do to that parameter?
3. §12 says curvature limits the learning rate. Our loss had constant curvature. What would go wrong if the curvature were huge in one place and tiny in another? (You have already met this in Chapter 2 §10, in different language.)
4. A numerical derivative needs two function evaluations per parameter. A modern model has $10^{11}$ parameters. How many evaluations is that per training step, and what does that tell you about why Chapter 27 exists?
5. If $f'(x) = 0$ everywhere on an interval, what can you say about $f$ there? Now: if $f''(x) = 0$ everywhere, what can you say?
6. We defined the derivative using $f(x+h)$ — a step to the *right*. Would using a step to the left give a different answer? Always, sometimes, or never — and what does your answer have to do with §13?

---

## 24. 🔭 Bridge to Chapter 07

We can now measure how fast a function changes, at a point, without dividing by zero. Chapter 1's compass is rigorous, its minus sign is explained, and even its mysterious learning-rate limit turned out to be curvature in disguise.

But look closely at what we actually differentiated. $L(w) = 7.5(w-2)^2$ is a function of **one** variable. To get it, Chapter 1 had to pin the bias at $b=1$ and study $w$ alone.

That was a convenience, and the real problem does not allow it. Our actual loss depends on $w$ **and** $b$ together — two dials, both free, both affecting the answer. The loss surface is not a curve you can walk along; it is a bowl you can move around in any direction.

So what is "the slope" of a bowl? Walk east and it might rise. Walk north and it might fall. Walk north-east and it does something in between. **There is no single number that answers "how steep is it here?"** — the question itself is incomplete until you say *in which direction*.

And a real network has not two parameters but a billion.

> **What is the slope of a function that depends on many variables at once — and which of the infinitely many directions should we step in?**

That is where Chapter 07 begins. It answers both halves, and the object it builds — the gradient — is the thing that actually trains every model in this course.

➡️ **Next:** [Chapter 07 — Partial Derivatives, Gradients and the Chain Rule](<../Lecture 07 - Partial Derivatives, Gradients and the Chain Rule/blog.md>)
