# Lecture 13 — Linear Regression, Properly

> **The Big Question:** What exactly are we assuming when we draw the “best-fit line,” and how does the data choose its parameters?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2013%20-%20Linear%20Regression%2C%20Properly/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

We just learned **Linear Regression, Properly**.

But using it creates a new question.

> **What problem does this idea still leave us unable to solve?**

That question leads naturally to **Logistic Regression: The One-Neuron Network**.

**Next → Logistic Regression: The One-Neuron Network.**

## 1. The Problem: Predict a House Price

Return to our house-price dataset:

| Rooms $x$ | Price $y$ (₹ lakh) |
|---:|---:|
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |
| 4 | 9 |

The pattern looks perfectly straight.

But imagine that real measurements are noisy:

| Rooms | Price |
|---:|---:|
| 1 | 3.2 |
| 2 | 4.8 |
| 3 | 7.1 |
| 4 | 8.9 |
| 5 | 11.4 |

Now we need more than “spot the pattern.” We need a model with parameters and a rule for choosing them.

Let

$$
\hat y = wx+b.
$$

The question is no longer *what does a line look like?*

It is:

> **Which $w$ and $b$ should our data choose?**

---

## 2. What Would a Solution Need?

A useful fitting method must:

1. Produce a prediction for every input.
2. Share one set of parameters across the dataset.
3. Penalize predictions that disagree with observations.
4. Have a reliable way to improve the parameters.
5. Make clear what assumptions turn the data into a particular fitting rule.

Chapter 12 has already given us the last clue: choose a probability model for the observations, then optimize its likelihood.

---

## 3. First Attempt: Connect the First and Last Points

A smart human might draw a line through the first and last observations.

For the noisy data above, that gives approximately

$$
w \approx \frac{11.4-3.2}{5-1}=2.05.
$$

Then

$$
b\approx3.2-2.05(1)=1.15.
$$

The line looks reasonable.

But it ignores three intermediate observations when choosing the parameters.

> ⚠️ **A Tempting Wrong Idea**
>
> *“A good regression line can be determined from any two points.”*
>
> Two points determine a line, but they do not determine the line that best explains **all** observations under a chosen error model.

We need a score that lets every observation vote.

---

## 4. The Discovery: Squared Error as the Objective

For every example,

$$
\text{error}_i=\hat y_i-y_i.
$$

Following Chapter 12's Gaussian-noise model, the natural objective is the sum of squared errors:

$$
\boxed{J(w,b)=\sum_{i=1}^{n}(wx_i+b-y_i)^2}
$$

Often we divide by $n$:

$$
\boxed{\mathrm{MSE}(w,b)=\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2}.
$$

The division changes the scale but not the minimizing $(w,b)$.

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Every house hands the line a report card. Square its mistake so positive and negative misses cannot cancel, then average the scores. |
| ✏️ **Numbers** | If errors are $+1,-2,+0.5$, the squared errors are $1,4,0.25$, so the average is $(1+4+0.25)/3=1.75$. |
| 🎓 **Abstraction** | $J(w,b)=\sum_i(wx_i+b-y_i)^2$ is a scalar surface over parameter space. |

---

## 5. Why Square the Error?

We should not pretend the square is compulsory.

The square appears here because of the Gaussian likelihood derived previously. But it has another useful property: a large error receives disproportionately large punishment.

For errors of magnitude 1 and 3:

$$
1^2=1,
\qquad
3^2=9.
$$

The three-times-larger miss contributes nine times as much squared error.

This is a **modeling choice with consequences**, not a mathematical law.

---

## 6. Deriving the Gradient

Now the parameters are two numbers, $w$ and $b$.

The loss is

$$
J(w,b)=\sum_i(wx_i+b-y_i)^2.
$$

Differentiate with respect to $w$:

$$
\frac{\partial J}{\partial w}
=
2\sum_i(wx_i+b-y_i)x_i.
$$

Why did $x_i$ appear?

Because the inner expression $wx_i+b-y_i$ changes by $x_i$ when $w$ changes.

Similarly,

$$
\frac{\partial J}{\partial b}
=
2\sum_i(wx_i+b-y_i).
$$

Put the two together:

$$
\boxed{
\nabla J(w,b)=
\begin{bmatrix}
2\sum_i(wx_i+b-y_i)x_i\\[4pt]
2\sum_i(wx_i+b-y_i)
\end{bmatrix}
}
$$

This is exactly the direction information gradient descent needs.

---

## 7. One Update by Hand

Take the noiseless four-point dataset:

$$
(x,y)=(1,3),(2,5),(3,7),(4,9).
$$

Start badly:

$$
w=0,\qquad b=0.
$$

Predictions are all zero, so the errors are

$$
-3,-5,-7,-9.
$$

For $w$:

$$
\frac{\partial J}{\partial w}
=2[(-3)(1)+(-5)(2)+(-7)(3)+(-9)(4)]
$$

$$
=2[-3-10-21-36]=-140.
$$

For $b$:

$$
\frac{\partial J}{\partial b}
=2[-3-5-7-9]=-48.
$$

With learning rate $\eta=0.01$:

$$
w_{new}=0-0.01(-140)=1.4
$$

$$
b_{new}=0-0.01(-48)=0.48.
$$

One step has moved the line toward the right answer $y=2x+1$.

> 🧠 **Think** — Why was the update for $w$ much larger than the update for $b$? The gradient with respect to $w$ is multiplied by the input $x_i$, so large inputs amplify that direction.

---

## 8. The Matrix Form

Writing the same calculation one example at a time becomes repetitive.

Add a column of ones for the bias:

$$
X=
\begin{bmatrix}
1 & x_1\\
1 & x_2\\
\vdots & \vdots\\
1 & x_n
\end{bmatrix},
\qquad
\boldsymbol\beta=
\begin{bmatrix}
 b\\w
\end{bmatrix}.
$$

Then predictions for all observations are one matrix product:

$$
\hat{\mathbf y}=X\boldsymbol\beta.
$$

The residual vector is

$$
\mathbf r=X\boldsymbol\beta-\mathbf y.
$$

And the loss becomes

$$
J(\boldsymbol\beta)=\mathbf r^T\mathbf r
=(X\boldsymbol\beta-\mathbf y)^T(X\boldsymbol\beta-\mathbf y).
$$

Expand it:

$$
J(\boldsymbol\beta)
=\boldsymbol\beta^TX^TX\boldsymbol\beta
-2\mathbf y^TX\boldsymbol\beta
+\mathbf y^T\mathbf y.
$$

Differentiate:

$$
\nabla J
=2X^TX\boldsymbol\beta-2X^T\mathbf y.
$$

At a stationary point,

$$
X^TX\boldsymbol\beta=X^T\mathbf y.
$$

These are the **normal equations**.

If $X^TX$ is invertible,

$$
\boxed{\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf y}.
$$

This closed-form solution is useful because it reveals the algebraic structure of least squares.

But it is not the universal training algorithm for machine learning. For large datasets and modern models, iterative optimization is usually more practical.

---

## 9. The Geometry: Projection

The matrix $X$ contains every direction the model is allowed to use.

The predictions $X\boldsymbol\beta$ therefore live in the **column space of $X$**.

Least squares asks for the point in that space closest to $\mathbf y$:

```text
            y
            •
           /|
          / |
         /  | residual
        /   |
-------•----+----------------  model space
      y_hat
```

The best prediction vector is the projection of the observed target vector onto the space the model can express.

The residual is perpendicular to that space at the optimum:

$$
X^T(\mathbf y-\hat{\mathbf y})=0.
$$

That single equation explains the normal equations geometrically.

---

## 10. What Does “Best” Mean?

This chapter has quietly used a phrase that needs discipline:

> **Best according to squared error under a Gaussian-noise model.**

Change the assumptions and the objective can change.

For example, absolute error gives

$$
\frac1n\sum_i|wx_i+b-y_i|,
$$

which corresponds to a different noise story and is less dominated by extreme errors.

So regression is not “draw the line that feels best.” It is:

```text
choose a model
   ↓
choose assumptions about observations
   ↓
derive likelihood
   ↓
derive loss
   ↓
fit parameters
```

---

## 🔬 The Experiment

In the notebook:

1. Start with the noiseless house data and predict the final slope.
2. Add a small amount of noise.
3. Fit the line using gradient descent.
4. Fit the same data using the normal-equation solution.
5. Compare the parameters and predictions.
6. Change one point into an extreme outlier and predict what squared error will do.

---

## How It Breaks

| Failure | Symptom | Cause |
|---|---|---|
| Fit through two points | ignores other observations | only two examples choose the line |
| Use an enormous learning rate | loss explodes or oscillates | optimization step is too large |
| Forget the bias column | line is forced through the origin | model cannot shift vertically |
| Compare raw loss across different dataset sizes | larger dataset looks “worse” automatically | sum scales with $n$ |
| Assume the closed form always works | matrix inverse fails | $X^TX$ may be singular or poorly conditioned |
| Ignore outliers | line moves strongly toward one extreme point | squaring amplifies large residuals |

---

## Shapes

With $n$ observations and one feature:

```text
X           : (n, 2)     # bias + feature
beta        : (2,)
y           : (n,)
y_hat       : (n,)
residual    : (n,)
loss        : scalar
X.T @ X     : (2, 2)
X.T @ y     : (2,)
```

Shapes are more than bookkeeping.

For example, $X^T\mathbf y$ must be a two-entry vector because there are two parameters: $b$ and $w$.

---

## 🎯 Machine Learning Connection

Linear regression is one of the simplest complete machine-learning systems:

- a representation: $x$
- a parameterized model: $wx+b$
- an observation model: Gaussian noise
- an objective: negative log-likelihood / squared error
- an optimizer: gradient descent or a closed-form solver

Later neural networks keep the same architecture of ideas while changing the model family.

---

## Distinctions That Matter

| Pair | Difference |
|---|---|
| Regression vs line drawing | Regression is a statistical estimation procedure; line drawing is only the visual output. |
| SSE vs MSE | MSE divides SSE by the number of observations. Same minimizer. |
| Closed-form vs gradient descent | One solves the least-squares equations directly; the other iteratively improves parameters. |
| Parameter vs prediction | $w,b$ define the model; $\hat y$ is the model's output for an input. |
| Residual vs true noise | A residual is observed after fitting; the true noise is a latent modeling assumption. |

---

## What We Discovered

1. Linear regression is a parameterized model, not a line someone simply draws.
2. Under Gaussian observation noise, maximum likelihood becomes squared-error minimization.
3. The gradient follows directly from differentiating the scalar loss.
4. All examples can be handled at once with matrix notation.
5. The normal equations are both algebraic and geometric: the residual is orthogonal to the model space.
6. Optimization method and statistical objective are separate ideas.

---

## Mathematics We Built

$$
\hat y=wx+b
$$

$$
\mathrm{MSE}=\frac1n\sum_i(wx_i+b-y_i)^2
$$

$$
\frac{\partial J}{\partial w}=2\sum_i(wx_i+b-y_i)x_i
$$

$$
\frac{\partial J}{\partial b}=2\sum_i(wx_i+b-y_i)
$$

$$
\hat{\mathbf y}=X\boldsymbol\beta
$$

$$
X^TX\boldsymbol\beta=X^T\mathbf y
$$

$$
\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf y
$$

---

## What Each Symbol Means

| Symbol | Read it as | Meaning | In code |
|---|---|---|---|
| $x_i$ | “x sub i” | feature for observation $i$ | `x[i]` |
| $y_i$ | “y sub i” | observed target | `y[i]` |
| $w$ | “weight” | slope parameter | `w` |
| $b$ | “bias” | intercept parameter | `b` |
| $\hat y_i$ | “y-hat” | prediction | `y_hat[i]` |
| $X$ | “capital X” | design matrix | `X` |
| $\boldsymbol\beta$ | “beta” | parameter vector | `beta` |
| $\mathbf r$ | “r” | residual vector | `residual` |

---

## One-Minute Explanation

Imagine every possible line as a pair of knobs: slope and intercept. For each choice, every house gets a prediction and therefore a mistake. Under a Gaussian-noise assumption, squared mistakes form the negative log-likelihood. The best line is the one with the smallest total squared mistake. Matrix notation lets us ask the same question for all observations at once, and the geometry says the final residual is perpendicular to every direction the model is allowed to express.

---

## Exercises

### Level 1 — Observe
Look at a fitted line and residual plot. Which observation has the largest influence and why?

### Level 2 — Calculate
For $w=2$, $b=1$, calculate predictions and squared errors for $(1,3)$, $(2,4)$ and $(3,8)$.

### Level 3 — Derive
Starting from $J(w,b)=\sum_i(wx_i+b-y_i)^2$, derive both partial derivatives.

### Level 4 — Investigate
Add one extreme outlier to the house data. Predict how the fitted slope changes, then run the experiment.

### Level 5 — Design
Design a loss for a setting where large errors should matter less than they do under squared error. State what observation/noise assumption would make your choice plausible.

---

## Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Saying regression “finds correlation” | Correlation is descriptive; regression estimates a predictive conditional relationship. |
| Calling the residual the noise | Noise is the modeled random component; residual is what remains after fitting. |
| Forgetting the intercept | It changes the model family and can create avoidable bias. |
| Treating least squares as universally robust | Squaring makes it sensitive to large outliers. |

---

## Socratic Questions

- Why does adding a bias parameter correspond to adding a column of ones?
- Why is the residual perpendicular to the model space at the optimum?
- When would absolute error be preferable to squared error?
- Why can a closed-form solution be impractical when the number of parameters becomes huge?

---

## 🔭 Bridge to Chapter 14

Our model can output any real number. But suppose the target is not a price.

> **What if the answer is “fraud or not fraud,” “disease or not disease,” or “spam or not spam”?**

A straight line can produce negative numbers and values larger than one. A probability cannot.

The next chapter asks how to keep the linear combination while turning it into a probability.