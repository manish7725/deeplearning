# Lecture 05 — Eigenvectors: What a Transformation Leaves Alone

> **The Big Question:** Which directions does a matrix leave pointing where they started, and what can we do once we know?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2005%20-%20Eigenvectors%3A%20What%20a%20Transformation%20Leaves%20Alone/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** A matrix is an action on space. Its columns say where the basis lands, its determinant says how area changes, and chaining matrices collapses into a single matrix.
**Today:** We hunt for the directions a transformation does not turn — and find that they expose the skeleton the matrix is built around.
**Next:** We can describe how a matrix changes space. We still cannot describe a *rate* of change, which is what learning actually needs.

---

## 1. The Problem: One Direction Refused to Move

Chapter 4 left an observation dangling. Apply the shear $S = \begin{bmatrix} 1&1\\0&1\end{bmatrix}$ to a few vectors and watch what happens to their **directions**:

$$
S\begin{bmatrix} 0\\1 \end{bmatrix} = \begin{bmatrix} 1\\1 \end{bmatrix}
\quad(\text{swung } 45°)
\qquad
S\begin{bmatrix} 1\\2 \end{bmatrix} = \begin{bmatrix} 3\\2 \end{bmatrix}
\quad(\text{swung too})
$$

Almost everything tilts. But:

$$
S\begin{bmatrix} 1\\0 \end{bmatrix} = \begin{bmatrix} 1\\0 \end{bmatrix}
$$

Nothing. That direction came out exactly as it went in.

The scaling matrix $\begin{bmatrix} 2&0\\0&3\end{bmatrix}$ does the same thing twice over — it leaves both axes pointing where they were and merely changes their lengths, one by a factor of 2 and the other by 3. Meanwhile a 90° rotation leaves *nothing* pointing where it started.

So matrices differ in a way Chapter 4 never captured. Some directions are **special to a particular matrix**: it acts on them by pure stretching, with no turning at all.

Why care? Because a transformation described as *"stretch by 2 along this direction and by 3 along that one"* is about as simple as a transformation can get. If every matrix could be described that way, matrices would stop being mysterious.

---

## 2. What Would an Answer Need?

We want a method that, given any matrix:

1. **Finds the special directions**, if there are any.
2. **Reports the stretch factor** for each one.
3. **Tells us honestly when there are none** — the rotation must not produce a fake answer.
4. **Lets us rebuild the matrix** from those directions, so the description is complete rather than partial.

---

## 3. First Attempt: Try Every Direction

The obvious method: take a vector, apply the matrix, check whether the output points the same way. Rotate the input a little and repeat.

$$
\begin{bmatrix} 1\\0 \end{bmatrix} \to \text{unchanged ✓}
\qquad
\begin{bmatrix} 1\\0.1 \end{bmatrix} \to \begin{bmatrix} 1.1\\0.1 \end{bmatrix} \;(\text{tilted ✗})
\qquad
\begin{bmatrix} 1\\0.2 \end{bmatrix} \to \begin{bmatrix} 1.2\\0.2 \end{bmatrix} \;(\text{tilted ✗})
$$

This is Chapter 4 §3's mistake again: there are infinitely many directions, and sampling them can only ever suggest an answer, never prove one. Worse, it would silently miss a special direction that happens to fall between our samples.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Search numerically until the output is close enough to parallel."*
>
> "Close to parallel" is not a property a direction either has or lacks — it depends on your tolerance. And the question is exact: either $A\mathbf{v}$ is a multiple of $\mathbf{v}$ or it is not. **Stop searching and write the condition down as an equation.**

---

## 4. The Discovery: Write the Question as an Equation

State the requirement in symbols. We want a vector $\mathbf{v}$ whose image is a scalar multiple of itself:

$$
\boxed{\;A\mathbf{v} = \lambda\mathbf{v}\;}
$$

That is the whole definition. $\mathbf{v}$ is an **eigenvector** — a direction the matrix does not turn. $\lambda$ (lambda) is its **eigenvalue** — the factor it gets stretched by. *Eigen* is German for "own" or "characteristic": these are the matrix's own directions.

One exclusion: $\mathbf{v} = \mathbf{0}$ satisfies the equation for every $\lambda$ and tells us nothing, so eigenvectors are required to be non-zero.

Now solve it. Move everything to one side — but carefully, because $A$ is a matrix and $\lambda$ is a number, so $A - \lambda$ is nonsense. Insert the identity matrix $I$ to fix the types:

$$
A\mathbf{v} - \lambda I\mathbf{v} = \mathbf{0}
\qquad\Longrightarrow\qquad
(A - \lambda I)\,\mathbf{v} = \mathbf{0}
$$

Read what this says: the matrix $(A - \lambda I)$ sends the non-zero vector $\mathbf{v}$ to the origin. Chapter 4 §8 gave that a name — $\mathbf{v}$ is in the **null space** — and told us exactly when a non-trivial null space exists:

$$
\det(A - \lambda I) = 0
$$

This is the **characteristic equation**, and it is a polynomial in $\lambda$ alone. The infinite search of §3 has become finding the roots of a polynomial.

> 💡 The whole move was refusing to hunt and instead writing the requirement as an equation. Chapter 4's determinant then converted "a direction survives" into "this number is zero."

> 📜 **History Lens — Euler, Cauchy, and a German Prefix**
>
> The idea arrived through physics, not algebra. In the 1750s Leonhard Euler, studying how rigid bodies rotate, found that every rotation of a solid body has an **axis** — a line that stays put while everything else turns. That is an eigenvector, three-quarters of a century before anyone wrote $A\mathbf{v} = \lambda\mathbf{v}$.
>
> In the 1820s Augustin-Louis Cauchy, working on quadratic forms and the "principal axes" of surfaces, produced the characteristic equation in essentially the form above and proved that symmetric matrices always have real roots — the fact that makes §10's PCA work at all.
>
> The name came last. Mathematicians called them "proper values" or "characteristic values" for decades; David Hilbert used *Eigenwert* in the early 1900s, and when the German literature was translated the prefix survived untranslated. So "eigenvalue" is a half-translated word — and its meaning, *the matrix's own value*, is exactly right.

---

## 5. Doing It by Hand

Take a matrix small enough to solve completely:

$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

**Step 1 — build $A - \lambda I$.**

$$
A - \lambda I = \begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix}
$$

**Step 2 — set its determinant to zero.** Using $ad - bc$ from Chapter 4 §7:

$$
(2-\lambda)(2-\lambda) - (1)(1) = 0
$$

**Step 3 — expand and solve.**

$$
\lambda^2 - 4\lambda + 4 - 1 = 0
\quad\Longrightarrow\quad
\lambda^2 - 4\lambda + 3 = 0
\quad\Longrightarrow\quad
(\lambda-3)(\lambda-1) = 0
$$

$$
\lambda_1 = 3, \qquad \lambda_2 = 1
$$

**Step 4 — find each direction.** For $\lambda_1 = 3$, solve $(A - 3I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2\end{bmatrix} = \begin{bmatrix} 0\\0\end{bmatrix}
\quad\Longrightarrow\quad
-v_1 + v_2 = 0
\quad\Longrightarrow\quad
v_1 = v_2
$$

So any vector along $(1,1)$ works. Both rows gave the same equation — that is the determinant being zero, showing up as a redundant row.

For $\lambda_2 = 1$: $v_1 + v_2 = 0$, giving the direction $(1,-1)$.

**Step 5 — check.**

$$
A\begin{bmatrix} 1\\1\end{bmatrix} = \begin{bmatrix} 3\\3\end{bmatrix} = 3\begin{bmatrix} 1\\1\end{bmatrix} \;\checkmark
\qquad
A\begin{bmatrix} 1\\-1\end{bmatrix} = \begin{bmatrix} 1\\-1\end{bmatrix} = 1\begin{bmatrix} 1\\-1\end{bmatrix} \;\checkmark
$$

The entire behaviour of $A$ on the whole plane is now one sentence: **stretch by 3 along the diagonal $(1,1)$, and leave the anti-diagonal $(1,-1)$ alone.**

Note that eigenvectors have no natural length — $(2,2)$ and $(-5,-5)$ are equally valid. Only the **direction** is determined, so we usually normalize to length 1.

---

## 6. When There Is No Answer

Requirement 3 said the method must fail honestly. Test it on the two awkward cases from §1.

**The shear** $S = \begin{bmatrix} 1&1\\0&1\end{bmatrix}$:

$$
\det\begin{bmatrix} 1-\lambda & 1 \\ 0 & 1-\lambda\end{bmatrix} = (1-\lambda)^2 = 0
\quad\Longrightarrow\quad
\lambda = 1 \text{ (twice)}
$$

Solving $(S-I)\mathbf{v} = \mathbf{0}$ gives $v_2 = 0$ — only the direction $(1,0)$. A repeated root, but just **one** eigenvector direction instead of two. Such matrices are called **defective**, and they cannot be rebuilt from their eigenvectors alone. Our §1 observation was right: the shear has exactly one special direction.

**The rotation** $R = \begin{bmatrix} 0&-1\\1&0\end{bmatrix}$:

$$
\det\begin{bmatrix} -\lambda & -1 \\ 1 & -\lambda \end{bmatrix} = \lambda^2 + 1 = 0
\quad\Longrightarrow\quad
\lambda^2 = -1
$$

No real solutions. And that is exactly correct — a 90° rotation turns *every* direction, so there is no real direction it leaves alone. The mathematics did not invent a fake answer; it reported the truth.

> ⚠️ **Precision, not false simplicity.** Over the complex numbers $\lambda = \pm i$, and those complex eigenvalues encode the rotation angle — genuinely useful in signal processing and in analysing recurrent networks (Chapter 40). But over the real plane, where our houses and pixels live, the honest answer is: none.

---

## 7. Diagonalization: Rebuilding the Matrix

Requirement 4 said the description should let us rebuild $A$. Put the eigenvectors in the columns of a matrix $P$ and the eigenvalues down the diagonal of $D$:

$$
P = \begin{bmatrix} 1 & 1 \\ 1 & -1\end{bmatrix},
\qquad
D = \begin{bmatrix} 3 & 0 \\ 0 & 1\end{bmatrix}
$$

Then

$$
\boxed{\;A = PDP^{-1}\;}
$$

For our matrix, $P^{-1} = \begin{bmatrix} 0.5 & 0.5\\ 0.5 & -0.5\end{bmatrix}$, and multiplying out gives back $\begin{bmatrix} 2&1\\1&2\end{bmatrix}$ exactly.

Read it right to left, as Chapter 4 §9 taught, and it tells a story in three acts:

| | | |
|---|---|---|
| $P^{-1}$ | **translate** | rewrite the vector in eigenvector coordinates |
| $D$ | **stretch** | scale each of those coordinates by its eigenvalue — the easy part |
| $P$ | **translate back** | express the result in ordinary coordinates again |

> 💡 **Intuition** — Every diagonalizable matrix is secretly just a scaling. It only *looks* complicated because we insist on describing it in the wrong coordinate system. Change to the matrix's own axes and all it does is stretch.

---

## 8. Why This Is Powerful: Repeated Application

Here is where eigenvalues stop being a curiosity. Suppose you need $A^{10}$ — apply the transformation ten times. Direct multiplication is nine matrix products. With diagonalization:

$$
A^2 = (PDP^{-1})(PDP^{-1}) = PD\underbrace{(P^{-1}P)}_{I}DP^{-1} = PD^2P^{-1}
$$

The inner $P^{-1}P$ cancels, and the pattern continues:

$$
A^{n} = PD^{n}P^{-1}
$$

And $D^n$ is trivial, because powering a diagonal matrix just powers each diagonal entry. For our $A$:

$$
D^{10} = \begin{bmatrix} 3^{10} & 0 \\ 0 & 1^{10}\end{bmatrix} = \begin{bmatrix} 59049 & 0 \\ 0 & 1 \end{bmatrix}
\quad\Longrightarrow\quad
A^{10} = \begin{bmatrix} 29525 & 29524 \\ 29524 & 29525\end{bmatrix}
$$

Now look at what happened to the two eigenvalues. $3^{10} = 59049$; $1^{10} = 1$. After ten applications, the $\lambda = 3$ direction is **59,049 times more important** than the other. Apply it a hundred times and the second direction is numerically invisible.

> 🎯 This single observation explains a startling amount of deep learning:
>
> - Repeatedly applying a matrix drives everything toward its **dominant eigenvector**. That is why the power-iteration experiment in the notebook converges.
> - If eigenvalues exceed 1, repeated application **explodes**; if they are below 1, it **vanishes**. That is precisely the exploding/vanishing gradient problem of Chapters 39–40, and the reason LSTMs were invented.
> - Chapter 4's Level-4 exercise asked what happens to the determinant of a long chain of matrices. This is the answer: products of eigenvalues, compounding.

---

## 9. Every Matrix, Not Just Square Ones: the SVD

Eigenvectors have two limitations. They need a **square** matrix — our data matrix $X$ is $4\times2$, so $X\mathbf{v} = \lambda\mathbf{v}$ does not even typecheck — and §6 showed some square matrices are defective anyway.

The fix is the **singular value decomposition**, which works for *every* matrix without exception:

$$
A = U\Sigma V^{T}
$$

Read right to left again: $V^T$ **rotates**, $\Sigma$ **stretches** along the new axes (its diagonal entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ are the **singular values**), and $U$ **rotates** again. So:

> **Every matrix, whatever its shape, is a rotation, then a stretch, then another rotation.** That is all any linear map has ever been.

For our four-house matrix $X$, the singular values are

$$
\sigma_1 \approx 2334.5, \qquad \sigma_2 \approx 1.23
$$

and that ratio should look familiar. Chapter 2 §10 found the loss landscape of this data had a condition number of about 3.6 million and was untrainable. Here is where that number came from:

$$
\left(\frac{\sigma_1}{\sigma_2}\right)^2 = (1898.6)^2 \approx 3{,}604{,}700
$$

The condition number of the loss surface is the **square of the ratio of the data's singular values**. Chapter 2 measured the symptom; this is the cause.

Because the singular values are ordered, keeping only the largest few gives the best possible approximation of $A$ using less information — the basis of image compression, recommender systems, and the low-rank adapters (LoRA) of Chapter 53.

---

## 10. 🔬 The Experiment: Finding the Axes of Real Data

Our four houses are four points in a plane. They are not scattered randomly — Chapter 2 §9 hinted that bigger houses have more rooms *and* more area. If that is true, the cloud of points has a natural long axis, and eigenvectors should find it.

That is **principal component analysis**: take the covariance matrix of the data and compute its eigenvectors. The dominant one is the direction of greatest variation.

> 🧠 **Predict before reading on.** Run PCA on rooms (values 2–4) and area (values 800–1600) as they are. Which direction comes out as "most important" — and is that a discovery about houses, or about units?

Run it on the raw data and the first component explains **100.00%** of the variance, pointing almost exactly along the area axis.

That is not a finding about houses. Area is measured in numbers ~400× larger than rooms, and variance is measured in squared units, so area's variance is ~140,000× larger. **PCA on unscaled data simply finds whichever feature has the biggest units.** It is Chapter 2 §10's lesson in a new costume: scale controls geometry.

Standardize both features first — subtract the mean, divide by the standard deviation — and repeat. The covariance matrix becomes the correlation matrix:

$$
C = \begin{bmatrix} 1 & 0.702 \\ 0.702 & 1 \end{bmatrix}
$$

whose eigenvalues are $1 + 0.702 = 1.702$ and $1 - 0.702 = 0.298$, with eigenvectors $(1,1)$ and $(1,-1)$.

Now the answer means something:

- **PC1** $= (1,1)/\sqrt{2}$ — rooms and area rising together. Call it *overall size*. It explains $1.702/2 = \mathbf{85.1\%}$ of the variation.
- **PC2** $= (1,-1)/\sqrt{2}$ — rooms up while area goes down. Call it *cramped versus spacious*. The remaining 14.9%.

Two measured features, and the data really lives along roughly one direction. That is what dimensionality reduction means, and it is why Chapter 46 can talk about a "1000-dimensional" representation that actually occupies far fewer directions.

---

## 11. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **Defective matrix** | fewer eigenvectors than dimensions | Repeated root with only one direction. Cannot diagonalize. §6 |
| **Complex eigenvalues** | no real answer | The matrix rotates. Correct, not a bug. §6 |
| **PCA on unscaled data** | the largest-unit feature "explains everything" | §10. Standardize first. |
| **Eigenvalues near 1 in a chain** | explode or vanish over many steps | $\lambda^n$ compounds. §8, and Chapters 39–40 |
| **Tiny $\sigma_{\min}$** | numerically unstable inverses | $\sigma_1/\sigma_n$ is the condition number. §9 |
| **Reading eigenvectors as causes** | confident nonsense | PC1 is the direction of most variance, which need not correspond to any real mechanism. |

---

## 12. 🎯 Machine Learning Connection

| Eigen-idea | Where it shows up |
|---|---|
| dominant eigenvalue of repeated maps | exploding/vanishing gradients (Ch 39–40) |
| condition number $\sigma_1/\sigma_n$ | why feature scaling decides trainability (Ch 2, 33) |
| eigenvalues of the loss Hessian | curvature, and the stability limit $\eta < 2/\lambda_{\max}$ (Ch 1, 8, 32) |
| PCA | dimensionality reduction, whitening, visualizing representations (Ch 46) |
| low-rank approximation | compression, and LoRA fine-tuning (Ch 53) |
| spectral properties of weight matrices | measuring what a trained network has learned (Ch 56) |

The stability threshold from Chapter 1 was an eigenvalue all along. We wrote $\eta < 2/\lambda_{\max}$ without being able to say what $\lambda_{\max}$ was; now you know — it is the largest eigenvalue of the loss surface's curvature, the steepest direction of the valley.

---

## 13. Distinctions That Matter

| | |
|---|---|
| **Eigenvector** — a direction unturned | **Eigenvalue** — how much that direction is stretched |
| **Eigen-decomposition** — square matrices, may fail | **SVD** — every matrix, always exists |
| **Eigenvalues of $A$** | **Singular values of $A$** — $\sigma_i^2$ are eigenvalues of $A^TA$ |
| **Diagonalizable** — a full set of eigenvectors | **Defective** — fewer than the dimension |
| **Variance explained** — a statistical fact | **Cause** — not implied by any of this |
| **Rank** (Ch 4) — how many directions survive | **Effective rank** — how many have non-negligible $\sigma$ |

---

## 14. What We Discovered

1. Some directions come out of a transformation pointing exactly where they went in, stretched but unturned.
2. Searching for them is hopeless; writing $A\mathbf{v} = \lambda\mathbf{v}$ and applying Chapter 4's determinant turns the search into a polynomial.
3. Solving that polynomial gives the stretch factors; the null space of $A - \lambda I$ gives the directions.
4. Not every matrix cooperates — shears are defective, rotations have no real eigenvectors, and the mathematics says so honestly.
5. When it does work, $A = PDP^{-1}$ says every such matrix is *just a scaling*, seen from the wrong coordinates.
6. Powers become trivial, and the dominant eigenvalue takes over exponentially — which is exactly why gradients explode or vanish.
7. The SVD extends all of it to every matrix: rotate, stretch, rotate.
8. The condition number that made Chapter 2's data untrainable is the square of the ratio of its singular values.
9. PCA is eigenvectors applied to covariance — and on unscaled data it merely finds the biggest units.

## 15. Mathematics We Built

$$
A\mathbf{v} = \lambda\mathbf{v}
\qquad
(A - \lambda I)\mathbf{v} = \mathbf{0}
\qquad
\det(A - \lambda I) = 0
$$

$$
A = PDP^{-1}
\qquad
A^{n} = PD^{n}P^{-1}
\qquad
A = U\Sigma V^{T}
$$

$$
C = \tfrac{1}{n-1}X_c^{T}X_c
\qquad
\text{variance explained by PC}_i = \frac{\lambda_i}{\sum_j \lambda_j}
$$

## 16. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $\lambda$ | eigenvalue — the stretch factor | `eigvals` |
| $\mathbf{v}$ | eigenvector — the unturned direction | `eigvecs[:, i]` |
| $I$ | identity — leaves everything alone | `np.eye(n)` |
| $\det(A-\lambda I)=0$ | "this direction survives" | `np.linalg.eig(A)` |
| $P$ | eigenvectors as columns | `eigvecs` |
| $D$ | eigenvalues on the diagonal | `np.diag(eigvals)` |
| $\sigma_i$ | singular value | `np.linalg.svd(A)[1]` |
| $\sigma_1/\sigma_n$ | condition number | `np.linalg.cond(A)` |
| PC1 | direction of greatest variance | top eigenvector of the covariance |

## 17. One-Minute Explanation

With no equations:

> A transformation turns almost every arrow. What is special about the few it does not turn, and why does that make applying the transformation a thousand times easy?

---

## 18. Exercises

**Level 1 — Observe.** Chapter 4's four transformations were scale, rotate, shear and project. Without calculating, say how many real eigenvector directions each one has, and what the eigenvalues should be. Then check one prediction: what must the eigenvalues of a projection be, given that applying it twice is the same as applying it once?

**Level 2 — Calculate (by hand).** Find the eigenvalues and eigenvectors of $B = \begin{bmatrix} 4&1\\2&3\end{bmatrix}$. Verify each by computing $B\mathbf{v}$ and checking it is $\lambda\mathbf{v}$. Then confirm two shortcuts on your answers: the eigenvalues should sum to the trace $4+3=7$, and multiply to $\det B = 10$. Why must those hold?

**Level 3 — Derive.** Prove that if $A\mathbf{v} = \lambda\mathbf{v}$ then $A^2\mathbf{v} = \lambda^2\mathbf{v}$, then extend it to $A^n$. Use it to prove that if every $|\lambda_i| < 1$, then $A^n\mathbf{x} \to \mathbf{0}$ for *every* starting $\mathbf{x}$ — and explain in one sentence what that means for a 50-layer network whose weight matrices all have small eigenvalues.

**Level 4 — Investigate** (notebook Steps 8–11). Start from a random vector and apply $A = \begin{bmatrix}2&1\\1&2\end{bmatrix}$ repeatedly, normalizing each time. Which direction does it converge to, and how many steps does it take? Then rerun with $A = \begin{bmatrix}2&0\\0&1.999\end{bmatrix}$. Convergence becomes dramatically slower — explain why using the ratio $\lambda_2/\lambda_1$.

**Level 5 — Design.** You have a $1000 \times 1000$ image and can store only 50 numbers per row. Using §9, design a compression scheme: what do you keep, what do you discard, and how do you reconstruct? Estimate the compression ratio. Then state the harder part — what kind of image would your scheme handle beautifully, and what kind would it ruin?

---

## 19. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "Every matrix has $n$ eigenvectors." | Defective matrices do not. §6 |
| "No real eigenvalues means I made an arithmetic error." | It usually means the matrix rotates. §6 |
| "Eigenvectors have a specific length." | Only the direction is determined; any non-zero multiple is the same eigenvector. §5 |
| "Eigenvalues and singular values are the same." | Equal only for symmetric positive-definite matrices. $\sigma_i^2$ are eigenvalues of $A^TA$. §9 |
| "PC1 is the most important variable." | It is a *combination* of variables, in the direction of most variance — and on unscaled data, of the largest unit. §10 |
| "PCA tells me what causes what." | It describes spread, not mechanism. §11 |

## 20. Socratic Questions

1. The trace of a matrix equals the sum of its eigenvalues and the determinant equals their product. What does that say about a matrix with a zero eigenvalue — and how does it connect to Chapter 4 §8?
2. A symmetric matrix always has real eigenvalues and perpendicular eigenvectors. Covariance matrices are always symmetric. Is that a lucky coincidence for PCA, or is it forced by what covariance means?
3. If repeatedly applying a matrix drives every vector toward the dominant eigenvector, what does that suggest about what a very deep network does to its inputs *before* we add activations?
4. Eigenvalues of the loss Hessian set the stability limit $\eta < 2/\lambda_{\max}$. What would you do if $\lambda_{\max}$ were enormous but $\lambda_{\min}$ were tiny — is one learning rate ever enough? (Chapter 32 is the answer.)
5. The SVD says every matrix is rotate–stretch–rotate. Where did the "shear" from Chapter 4 §6 go? It is not a rotation and not a scaling.
6. PCA found that our houses vary mostly along one direction. If you kept only PC1, what exactly would you lose about house C — and could you tell you had lost it?

---

## 21. 🔭 Bridge to Chapter 06

Part I is finished. We can now say what a number, a list, a table and a transformation are, and we have taken a matrix apart to see the directions it is built around.

Every one of those tools describes **structure that sits still**. A vector is at a place. A matrix sends a point somewhere. An eigenvector is a direction that stays.

But Chapter 1 did not need any of that. It needed something else entirely, and we faked it. When we asked *"which way should I nudge $w$ to reduce the loss?"*, we computed the loss at two nearby values, took a difference, divided by the gap, and watched what happened as the gap shrank to nothing. We called the result a slope and moved on, promising to come back.

That promise is due. The quantity we invented is not about where things *are* — it is about how fast one thing changes when another thing moves, measured at a single instant. None of Part I can express that.

> **What does it mean to measure a rate of change at a single point, where nothing has had a chance to change yet?**

That is where Chapter 06 begins, and it makes Chapter 1's borrowed compass rigorous at last.

➡️ **Next:** [Chapter 06 — Derivatives: The Compass for Learning](<../Lecture 06 - Derivatives: The Compass for Learning/blog.md>)
