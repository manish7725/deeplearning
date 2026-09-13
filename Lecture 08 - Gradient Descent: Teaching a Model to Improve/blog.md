# Lecture 08 — Gradient Descent: Teaching a Model to Improve

> **The Big Question:** We know which way is downhill. How far do we step, how many times, and does repeating it actually arrive anywhere?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2008%20-%20Gradient%20Descent%3A%20Teaching%20a%20Model%20to%20Improve/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We now have a compass: the gradient tells us which way the loss rises fastest.

So the machine has a question it can finally answer: **which way should I move to make the loss smaller?**

> **Can we turn the gradient into a simple repeatable learning process?**

**Next → Gradient Descent.**

## 1. The Problem: A Compass Is Not a Journey

Stand at $w_1 = 0$, $b = 0$ on our house-price loss. Chapter 7 computed the gradient there:

$$
\nabla L = \begin{bmatrix} -38.5 \\ -13.0 \end{bmatrix}
\qquad
L(0,0) = 45.0
$$

Both components are negative, so both dials should increase. Fine — increase them by how much?

Try following the compass literally and stepping the full length of the gradient:

$$
\begin{bmatrix} w_1 \\ b \end{bmatrix}
\leftarrow
\begin{bmatrix} 0 \\ 0 \end{bmatrix} - \begin{bmatrix} -38.5 \\ -13.0 \end{bmatrix}
= \begin{bmatrix} 38.5 \\ 13.0 \end{bmatrix}
$$

The true answer is $w_1 = 2$, $b = 1$. We wanted to travel a distance of about $2.2$; we have landed at $(38.5, 13)$, roughly **forty** units away, and the loss there is not 45 but $13{,}544$. We aimed downhill and ended up three hundred times higher than we started.

> 🧠 **Think** — The gradient told the truth. It said *"from right here, the loss falls fastest in this direction."* It said nothing about how far that remains true, and it could not — it is a statement about an instant, and we took a journey.

So the compass alone is not enough. Three questions block us:

1. **How far** should a single step be?
2. **How many** steps, and how do we know when to stop?
3. **Does repeating this even work?** Nothing so far proves that following a local direction over and over arrives anywhere at all, rather than circling forever or wandering off.

---

## 2. What Would a Solution Need?

1. **Use only local information.** At a billion parameters we can compute $\nabla L$ at our current point and essentially nothing else. No map of the whole landscape.
2. **Make progress every step**, or at least not go backwards.
3. **Arrive.** Repeated application must actually reach the bottom, not merely point at it.
4. **Tell us when it will fail** — before we waste a week of compute finding out.
5. **Scale.** The same rule must work for 2 parameters and for $10^{11}$.

Requirement 4 is unusual, and by §7 we will be able to predict the exact learning rate at which our own training blows up.

---

## 3. First Attempt: Jump Straight to the Bottom

Before iterating, consider the obvious alternative. At the minimum the gradient is zero — Chapter 7 confirmed both partials vanish at $(2,1)$. So why not simply **solve** $\nabla L = \mathbf{0}$ and go there directly?

For our loss we can. Setting both partials to zero:

$$
16.5(w_1-2) + 5.5(b-1) = 0
\qquad
5.5(w_1-2) + 2(b-1) = 0
$$

Two linear equations, two unknowns. Solving gives $w_1 = 2$, $b = 1$ exactly, in one shot, with no iteration and no learning rate. It works perfectly.

And it is a dead end. Here is why.

**It needs the loss to be a quadratic.** Our loss is a parabola-bowl only because the model is linear and the loss is squared error. Put a single activation function in the model — Chapter 25 does — and $\nabla L = 0$ becomes a system of equations with no closed-form solution at all. There is nothing to solve.

**Even when solvable, it does not scale.** Solving $n$ linear equations costs roughly $n^3$ operations. For $n = 2$ that is 8. For a small network with $n = 10^6$ it is $10^{18}$ operations — more than a modern GPU can do in a year. For $n = 10^{11}$ it is beyond absurd.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Just set the gradient to zero and solve — iteration is a workaround for people who cannot do algebra."*
>
> It is the reverse. Iteration is what survives when algebra runs out. Direct solution is a luxury available for exactly one family of models, and we will use it — Chapter 13 derives the normal equations and uses them as ground truth to check gradient descent against. But the algorithm that trains every modern network cannot be that one.

We need a method that uses only the local gradient, and gets there by repetition.

---

## 4. The Discovery: Small Steps, Repeated

The failure in §1 was not the direction. It was the *length*. So keep the direction and shrink the step:

$$
\boxed{\;\theta \;\leftarrow\; \theta - \eta\,\nabla L(\theta)\;}
$$

The number $\eta$ — the **learning rate** — is the fraction of the gradient we actually walk. This is **gradient descent**, and it is the entire algorithm. Everything after this in the chapter is analysis of when it works.

Watch it run from $(0,0)$ with $\eta = 0.05$:

| step | $w_1$ | $b$ | $L$ |
|---:|---:|---:|---:|
| 0 | $0.0000$ | $0.0000$ | $45.0000$ |
| 1 | $1.9250$ | $0.6500$ | $0.3133$ |
| 2 | $2.0831$ | $0.7056$ | $0.0091$ |
| 3 | $2.0955$ | $0.7122$ | $0.0069$ |
| 5 | $2.0952$ | $0.7169$ | $0.0067$ |
| 10 | $2.0917$ | $0.7273$ | $0.0062$ |
| 20 | $2.0851$ | $0.7471$ | $0.0053$ |
| 220 | $2.0189$ | $0.9438$ | $0.000263$ |

Look carefully, because the shape of this is the whole lesson. **Two steps** take the loss from 45 to 0.009 — a factor of five thousand. And then it very nearly stops: over the next two hundred steps the loss improves by only another factor of thirty.

Watch the two columns separately and you can see why. $w_1$ arrives almost immediately — it is at $2.08$ after two steps and barely moves again. Meanwhile $b$ is at $0.71$ after three steps and is *still* only at $0.94$ two hundred steps later, inching towards its target of 1.

One dial sprints; the other crawls. **That is not a flaw in the code, and §6 explains exactly what causes it.**

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Walking down a foggy hillside. You can see only the ground at your feet, so you take a short step downhill, look again, and repeat. You never need a map of the mountain. |
| ✏️ **Numbers** | At $(0,0)$: $\nabla L = [-38.5, -13.0]$, so with $\eta=0.05$ we move by $[+1.925, +0.65]$ — a step of the right size, not forty times too big. |
| 🎓 **Abstraction** | $\theta_{t+1} = \theta_t - \eta\nabla L(\theta_t)$, iterated until the gradient is small. |

> 📜 **History Lens — Cauchy, 1847**
>
> The method is older than almost everything it now trains. **Augustin-Louis Cauchy** — the same Cauchy who made Chapter 6's limits rigorous and who wrote down Chapter 5's characteristic equation — published it in 1847, in a short note on solving systems of simultaneous equations. His actual problem was astronomical: fitting orbital calculations to observations, where the equations were too tangled to solve directly.
>
> His idea was exactly §4: follow the steepest descent of a quantity you want to minimize, one step at a time.
>
> It then sat mostly unused for a century, because by hand each step is laborious and thousands are needed. The algorithm was waiting for a machine that could repeat arithmetic without getting bored. **Gradient descent is a 19th-century idea that had to wait for the 20th century's hardware** — and it now runs on essentially every model in this course.

---

## 5. Does It Arrive? The Contraction Argument

Requirement 3 demanded proof, not hope. For our loss we can give one, and it is worth every line because it explains *all* of the behaviour we see.

Write the error — how far we currently are from the answer — as

$$
\mathbf{e}_t = \theta_t - \theta^\star
$$

where $\theta^\star = (2, 1)$ is the bottom. Chapter 7 showed that for our quadratic loss the gradient is exactly

$$
\nabla L(\theta) = H\,(\theta - \theta^\star), \qquad
H = \begin{bmatrix} 16.5 & 5.5 \\ 5.5 & 2 \end{bmatrix}
$$

where $H$ is the **Hessian**, the matrix of second derivatives — the multi-dimensional curvature from Chapter 7 §8. Substitute that into the update:

$$
\theta_{t+1} = \theta_t - \eta H(\theta_t - \theta^\star)
$$

Subtract $\theta^\star$ from both sides to get the next error:

$$
\mathbf{e}_{t+1} = \mathbf{e}_t - \eta H\mathbf{e}_t = (I - \eta H)\,\mathbf{e}_t
$$

**The error is multiplied by a fixed matrix every step.** So after $t$ steps,

$$
\mathbf{e}_t = (I - \eta H)^{t}\,\mathbf{e}_0
$$

and the entire question "does it arrive?" becomes "does $(I-\eta H)^t$ shrink to nothing?" — which is precisely the question Chapter 5 §8 answered about repeated application of a matrix.

### Now use the eigenvectors

Chapter 5 taught that a symmetric matrix is just stretching along its eigenvector directions. $H$ is symmetric, and its eigenvalues are

$$
\lambda_{\max} = 18.3501, \qquad \lambda_{\min} = 0.1499
$$

Decompose the error into those two directions. Along an eigendirection with eigenvalue $\lambda$, the matrix $(I - \eta H)$ simply multiplies by the number $(1 - \eta\lambda)$. So that piece of the error behaves as

$$
e_t = (1 - \eta\lambda)^{t}\,e_0
$$

A number raised to the power $t$. That shrinks to zero **if and only if** its size is below 1:

$$
|1 - \eta\lambda| < 1
\qquad\Longleftrightarrow\qquad
0 < \eta < \frac{2}{\lambda}
$$

Every eigendirection must shrink, so the binding constraint is the largest eigenvalue:

$$
\boxed{\;\eta < \frac{2}{\lambda_{\max}}\;}
$$

For our loss:

$$
\eta < \frac{2}{18.3501} = 0.108991
$$

This is Chapter 6's one-variable rule $\eta < 2/L''$, generalized. With one parameter the curvature was one number; with many it is a matrix, and the ceiling is set by its steepest direction — because a single $\eta$ has to survive the worst case.

> 🧪 The notebook runs it: $\eta = 0.108$ converges, $\eta = 0.109$ does not. The prediction is right to three decimal places, and it was made **before** running anything.

---

## 6. Why It Slows Down

The same argument explains the crawl in §4's table, and this is the part most people never get told.

Convergence is not governed by one number but by *all* of them. The fast directions die quickly; the slow ones set how long you wait. At $\eta = 0.05$:

| direction | $\lambda$ | contraction per step $\lvert 1-\eta\lambda\rvert$ | after 200 steps |
|---|---:|---:|---:|
| steep | $18.3501$ | $0.0825$ | $\approx 10^{-217}$ — gone instantly |
| flat | $0.1499$ | $0.9925$ | $0.22$ — barely touched |

The steep direction is annihilated in a handful of steps. The flat direction shrinks by three-quarters of one percent per step, and *that* is what you are waiting for.

This is precisely the two-speed behaviour in §4's table. $w_1$ lies mostly along the steep eigendirection, so it arrives in two steps and stops moving. $b$ lies mostly along the flat one, so it is still creeping from $0.75$ towards $1$ after two hundred steps. The algorithm is not "converging slowly" — it converged in one direction almost instantly and is waiting on the other.

The ratio of the two eigenvalues has a name we already know — Chapter 5 called it the **condition number**:

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{18.3501}{0.1499} = 122.4
$$

It measures how badly stretched the bowl is. And it controls the speed directly. Even with the **best possible** learning rate, the error can shrink no faster than

$$
\rho = \frac{\kappa - 1}{\kappa + 1} = \frac{121.4}{123.4} = 0.9838
$$

per step — about 1.6% per step, needing roughly 42 steps to halve the error and 850 to reach six decimal places. That is the *ceiling on performance* for this problem, not a property of our code.

> ⚠️ **Precision, not false simplicity.** The learning rate minimizing that worst-case rate is $\eta^\star = 2/(\lambda_{\max}+\lambda_{\min}) = 0.1081$. But "optimal" here means *optimal against the worst starting point*. From our particular start at $(0,0)$, $\eta = 0.10$ actually finishes closer after 200 steps than $\eta^\star$ does, because most of our initial error happens to lie along the steep direction, which $\eta = 0.10$ kills faster. Worst-case optimal is not the same as best-on-this-run.

And this is Chapter 2's lesson returning with a number attached. Chapter 2 found that unscaled features gave a condition number in the millions and made training impossible. Now you can see the mechanism: $\kappa$ is exactly the ratio that appears in $\rho$, and as $\kappa$ grows, $\rho \to 1$ and progress stops.

> 🎯 **Everything in Chapter 32 — momentum, RMSProp, Adam — exists to beat this $\rho$.** They are not arbitrary tricks; they are attempts to escape a bound that plain gradient descent cannot.

---

## 7. 🔬 The Experiment: Four Regimes of the Learning Rate

> 🧠 **Predict before reading on.** We run from $(0,0)$ for 200 steps with $\eta = 0.02$, $0.05$, $0.10$, $0.108$, $0.109$ and $0.12$. Theory says the ceiling is $0.108991$. Which runs land at $(2,1)$, and does failure look like a wrong answer or something else?

Measuring the distance to the true minimum after 200 steps:

| $\eta$ | distance to $(2,1)$ | verdict |
|---:|---:|---|
| $0.02$ | $0.170$ | converging, too slowly |
| $0.05$ | $0.0689$ | converging |
| $0.10$ | $0.0151$ | converging well |
| $0.108$ | $0.0576$ | converging, just under the ceiling |
| $0.109$ | $2.289$ | **past the ceiling — moving away** |
| $0.12$ | overflow | diverged |

The theory said $0.108991$. The behaviour changes between $0.108$ and $0.109$.

Two things are worth noticing. First, failure is not a wrong answer — it is **growth**. Above the ceiling each step overshoots the valley and lands higher on the opposite wall, and the overshoot compounds geometrically, exactly as $(1-\eta\lambda)^t$ with $|1-\eta\lambda| > 1$ predicts.

Second, $\eta = 0.02$ "works" and is still useless — after 200 steps it is ten times further away than $\eta = 0.10$. **Slowness is a failure mode too**, and it is the more dangerous one, because nothing crashes to tell you.

---

## 8. What the Bottom Actually Looks Like

Requirement 3 said "arrive". Arrive where, exactly?

Gradient descent stops moving when $\nabla L = \mathbf{0}$. Such a point is called **stationary**, and Chapter 6 §10 already warned that stationary does not mean minimum. In many dimensions there are three kinds:

**A local minimum** — curving up in every direction. The algorithm settles here and cannot leave, because every direction is uphill.

**A local maximum** — curving down in every direction. Gradient descent will never stop here in practice; the tiniest nudge falls away.

**A saddle point** — up in some directions, down in others. The classic is

$$
f(x,y) = x^2 - y^2
$$

At the origin $\nabla f = \mathbf{0}$, so the algorithm has no reason to move. But it is a minimum along $x$ and a **maximum** along $y$ — the shape of a horse's saddle or a mountain pass. The gradient is zero and yet you are not at the bottom of anything.

### Local minima, concretely

Our house-price loss is a single clean bowl, which is a special property of linear models with squared error — it is **convex**, with exactly one minimum. Most losses are not. Take

$$
f(x) = x^4 - 8x^2
$$

Its derivative $f'(x) = 4x^3 - 16x = 4x(x^2-4)$ vanishes at $x = -2, 0, +2$. The second derivative $f''(x) = 12x^2 - 16$ classifies them: $f''(0) = -16 < 0$ is a maximum, and $f''(\pm2) = +32 > 0$ are two separate minima, both with $f = -16$.

Run gradient descent on it and where you land depends entirely on where you start:

| start | ends at |
|---:|---:|
| $-3.0$ | $-2$ |
| $-0.5$ | $-2$ |
| $+0.5$ | $+2$ |
| $+3.0$ | $+2$ |

**The algorithm has no memory and no map.** It goes downhill from wherever it happens to be, and different starting points reach different answers. This is why initialization matters (Chapter 33), and why training the same model twice with different random seeds gives different results.

> 💡 A reassurance worth stating honestly: for very large networks, the practical difficulty is usually *not* getting stuck in bad local minima. High-dimensional loss surfaces have far more saddle points than local minima, and most minima that are found turn out to be of similar quality. Chapter 51 examines the evidence for this properly rather than asserting it.

---

## 9. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **$\eta$ above $2/\lambda_{\max}$** | loss grows, then overflow | $\lvert 1-\eta\lambda\rvert > 1$, compounding. §5 |
| **$\eta$ far too small** | loss falls, forever | Contraction near 1. Nothing crashes; you just wait. §7 |
| **Large condition number** | fast start, endless crawl | $\rho = (\kappa-1)/(\kappa+1) \to 1$. §6 |
| **Saddle point** | gradient tiny, loss stuck, not a minimum | Zero gradient is necessary, not sufficient. §8 |
| **Bad initialization** | different answer every run | Non-convex losses have many basins. §8 |
| **Stopping on gradient size alone** | stopping at a saddle or a plateau | Small gradient does not mean good solution. |

---

## 10. 🎯 Machine Learning Connection

Everything in this chapter is what "training" means. When a log line reads `epoch 3, loss 0.42`, that is §4 having run a few thousand times.

| This chapter | In a real system |
|---|---|
| $\theta \leftarrow \theta - \eta\nabla L$ | `optimizer.step()` |
| $\eta$ | the single most-tuned hyperparameter in deep learning |
| $\eta < 2/\lambda_{\max}$ | why your loss suddenly became `nan` |
| $\kappa$, the condition number | why we normalize inputs (Ch 2) and activations (Ch 33) |
| $\rho = (\kappa-1)/(\kappa+1)$ | the bound Adam and momentum exist to beat (Ch 32) |
| saddle points, local minima | why seeds change results (Ch 33, Ch 51) |
| computing $\nabla L$ | the hard part — Chapters 27, 28 and 31 |

One honest gap: we computed $\nabla L$ from a formula we derived by hand for a two-parameter linear model. No one derives gradients by hand for a network with a hundred layers. **How to compute $\nabla L$ automatically, for any model, cheaply** is the subject of Chapters 27–31, and it is the single piece of machinery that made deep learning practical.

---

## 11. Distinctions That Matter

| | |
|---|---|
| **Gradient** — the direction (Ch 7) | **Gradient descent** — the algorithm that uses it |
| **Direct solution** — exact, needs a quadratic, costs $n^3$ | **Iteration** — approximate, works on anything, costs $n$ per step |
| **Diverging** — loss grows without bound | **Crawling** — loss falls, uselessly slowly |
| **Stationary point** — $\nabla L = 0$ | **Minimum** — stationary *and* curving up everywhere |
| $\lambda_{\max}$ — sets the ceiling on $\eta$ | $\lambda_{\min}$ — sets how long you wait |
| **Convex loss** — one basin, any start works | **Non-convex** — many basins, the start decides |
| $\eta$ — chosen by you | $\theta$ — learned from data |

---

## 12. What We Discovered

1. A compass is not a journey: stepping the full gradient overshot the answer by a factor of forty.
2. Solving $\nabla L = 0$ directly works for our loss and is a dead end — it needs a quadratic, and costs $n^3$.
3. Keeping the direction and shrinking the step gives gradient descent, the whole algorithm in one line.
4. For a quadratic the error obeys $\mathbf{e}_{t+1} = (I - \eta H)\mathbf{e}_t$, so convergence is Chapter 5's question about repeated application of a matrix.
5. Along each eigendirection the error is multiplied by $(1-\eta\lambda)$ every step, which shrinks exactly when $\eta < 2/\lambda$.
6. The ceiling is therefore $\eta < 2/\lambda_{\max} = 0.108991$, and experiment breaks between $0.108$ and $0.109$.
7. The *smallest* eigenvalue sets the waiting time, so the condition number $\kappa = 122$ explains the fast start and long crawl.
8. Even optimally tuned, the rate is bounded by $(\kappa-1)/(\kappa+1) = 0.9838$ — the bound Chapter 32's optimizers exist to beat.
9. Zero gradient means stationary, not minimal: saddle points and local maxima also qualify.
10. On non-convex losses the starting point decides the answer, which is why seeds matter.

## 13. Mathematics We Built

$$
\theta_{t+1} = \theta_t - \eta\,\nabla L(\theta_t)
$$

$$
\nabla L(\theta) = H(\theta - \theta^\star)
\qquad
\mathbf{e}_{t+1} = (I - \eta H)\,\mathbf{e}_t
\qquad
\mathbf{e}_t = (I-\eta H)^t\mathbf{e}_0
$$

$$
e_t = (1-\eta\lambda)^t e_0
\qquad
|1-\eta\lambda| < 1
\qquad
\eta < \frac{2}{\lambda_{\max}}
$$

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}
\qquad
\eta^\star = \frac{2}{\lambda_{\max}+\lambda_{\min}}
\qquad
\rho = \frac{\kappa-1}{\kappa+1}
$$

## 14. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $\theta$ | all parameters together | `params` |
| $\eta$ | learning rate — the fraction of the gradient walked | `learning_rate` |
| $\theta^\star$ | the parameters at the bottom | — |
| $\mathbf{e}_t$ | error — how far we still are | `theta - theta_star` |
| $H$ | Hessian — the curvature matrix | `hessian` |
| $\lambda_{\max}, \lambda_{\min}$ | steepest and flattest curvature | `np.linalg.eigvalsh(H)` |
| $\kappa$ | condition number | `np.linalg.cond(H)` |
| $\rho$ | best achievable contraction per step | — |
| $I$ | identity matrix | `np.eye(n)` |

## 15. One-Minute Explanation

With no equations:

> You are on a foggy hillside and can see only the ground at your feet. Describe how you get to the bottom — and explain the two completely different ways that plan can fail.

---

## 16. Exercises

**Level 1 — Observe.** Look at §4's table. The loss falls from 45 to 1.2 in one step, then takes 199 more steps to remove the remaining 1.2. Which eigendirection was killed in that first step, and which one are we waiting for? Now look at §7: $\eta = 0.02$ and $\eta = 0.109$ both fail. Describe the difference in how they fail, and say which is more dangerous in practice.

**Level 2 — Calculate (by hand, no code).** Starting at $(w_1, b) = (0,0)$ with $\eta = 0.05$, use $\partial L/\partial w_1 = 16.5(w_1-2)+5.5(b-1)$ and $\partial L/\partial b = 5.5(w_1-2)+2(b-1)$ to compute the first **two** steps by hand. Then evaluate $L = 8.25u^2 + 5.5uv + v^2$ (with $u = w_1-2$, $v = b-1$) at each point and confirm the loss went $45 \to 0.3133 \to 0.0091$. Finally, explain why $w_1$ has essentially arrived after two steps while $b$ is still only at $0.71$.

**Level 3 — Derive.** Starting from $\theta_{t+1} = \theta_t - \eta H(\theta_t - \theta^\star)$, derive $\mathbf{e}_{t+1} = (I-\eta H)\mathbf{e}_t$ in full. Then, for a single eigendirection, show the error is $(1-\eta\lambda)^t e_0$ and deduce the condition $0 < \eta < 2/\lambda$. Finally prove that the $\eta$ minimizing $\max\big(|1-\eta\lambda_{\max}|,\,|1-\eta\lambda_{\min}|\big)$ is $\eta^\star = 2/(\lambda_{\max}+\lambda_{\min})$. (Hint: at the optimum the two quantities are equal and of opposite sign.)

**Level 4 — Investigate** (notebook Steps 8–11). Verify the contraction factors: run at $\eta = 0.05$, decompose the error onto the two eigenvectors of $H$ at each step, and check each component shrinks by exactly $|1-\eta\lambda|$ per step. Then find empirically the $\eta$ that gets closest to the optimum after 200 steps from $(0,0)$, and explain why it is **not** $\eta^\star = 0.1081$ — what does your starting point have to do with it?

**Level 5 — Design.** Gradient descent takes one step size in every direction, so §6's ceiling and crawl are unavoidable. Design a rule that takes **large** steps along flat directions and **small** steps along steep ones. What extra information would it need? Estimate its cost per step for $n$ parameters, and say why that cost rules it out for $n = 10^{11}$. Then propose a cheap approximation that keeps most of the benefit — you are inventing the idea behind Chapter 32's adaptive optimizers.

---

## 17. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "A bigger learning rate means faster learning." | Only up to $2/\lambda_{\max}$. Past it you move *away* from the answer. §5 |
| "The loss went down, so $\eta$ is fine." | It may be a hundred times too small. Slowness does not announce itself. §7 |
| "It converged, so we are at the global minimum." | Only guaranteed for convex losses. §8 |
| "Zero gradient means minimum." | Saddle points and maxima have zero gradient too. §8 |
| "Iteration is a workaround for not solving it exactly." | Direct solution needs a quadratic and costs $n^3$. §3 |
| "Condition number is a numerical-analysis detail." | It is the difference between 40 steps and 40,000. §6 |

## 18. Socratic Questions

1. The update uses only $\nabla L$ at the current point. What information about the landscape is it throwing away, and which chapter do you think goes back for it?
2. We proved convergence for a quadratic loss. Real losses are not quadratic. Does the proof still tell you anything useful near a minimum — and why might "near" be the only place that matters?
3. If $\eta = 2/\lambda_{\max}$ exactly, what is $|1-\eta\lambda_{\max}|$? What does the error do along that direction — shrink, grow, or something else?
4. §6 says even optimal tuning is bounded by $(\kappa-1)/(\kappa+1)$. What would $\kappa = 1$ mean geometrically, and how many steps would descent need then?
5. Saddle points have zero gradient but are not minima. If you could only compute gradients and not curvature, how would you ever tell you were sitting on one?
6. We stop when the gradient is small. Name a situation where that is exactly the wrong stopping rule.

---

## 19. 🔭 Bridge to Chapter 09

We now have a complete training algorithm. We know its update rule, we can prove when it converges, we can predict the exact learning rate at which it explodes, and we understand why it crawls. Given a loss function, we can minimize it.

**Given a loss function.**

Look back at what we have actually been minimizing, all the way from Chapter 1: the mean of the squared differences between prediction and truth. Chapter 1 was honest about where that came from — it said squaring was a *choice*, made because it removes signs and punishes big mistakes more than small ones, and it explicitly admitted that mean absolute error would have been just as defensible. Then we built eight chapters of machinery on top of that choice and never revisited it.

That should bother you. We have a rigorous method for finding the minimum of a function we picked by taste.

And there is a second thing we have quietly assumed. Our four houses sit *exactly* on a line — that is why the loss reaches precisely zero. Real house prices do not. Two identical houses on the same street sell for different amounts, because of the weather, the buyer's mood, a leaking tap noticed on the viewing. Real data **scatters**, and a model that drives its training loss to zero on scattered data is fitting noise.

So before we can justify a loss, we need a language for describing scatter: what a typical value is, how spread out the values are, and what "noise" even means as a mathematical object.

> **Real measurements scatter. How do we describe that scatter — and could describing it properly tell us which loss function we should have been using all along?**

That is where Chapter 09 begins. Four chapters later, Chapter 12 will derive squared error from first principles — and show that it was never really a choice at all, but a consequence of an assumption about noise we had been making without knowing it.

➡️ **Next:** [Chapter 09 — Describing Data: Mean, Variance, Distributions](<../Lecture 09 - Describing Data: Mean, Variance, Distributions/blog.md>)
