# Lecture 04 — A Matrix Can Transform Space

> **The Big Question:** What does a matrix *do*, and does chaining many of them build something new?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2004%20-%20A%20Matrix%20Can%20Transform%20Space/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** A matrix held an entire dataset, and $XW$ answered every question about every house in one operation.
**Today:** We notice that $XW$ did not merely *store* numbers — it moved every point out of one space and into another. We find out what that movement actually is.
**Next:** Most directions get twisted by a transformation. A few survive pointing exactly where they started, and those turn out to matter enormously.

---

## 1. The Problem: What Did the Matrix Actually Do?

At the end of Chapter 3 every house went in as a point with two coordinates — rooms and area — and came out as a point with three: market, insurance, tax. We called it filing. It was not filing.

Take the simplest possible version. A matrix, and a point:

$$
A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix},
\qquad
\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix},
\qquad
A\mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

The point was at $(1,1)$. Now it is at $(2,3)$. It *moved*. And the same matrix moves every other point too — $(1,0)$ goes to $(2,0)$, $(0,1)$ goes to $(0,3)$, $(5,5)$ goes to $(10,15)$.

So a matrix is not a filing cabinet. It is a **machine that relocates every point in space at once.**

That reframing makes three questions urgent, and none of them can be answered by the spreadsheet picture:

1. What *kinds* of movement are possible? Can a matrix stretch, rotate, flatten?
2. If I apply $A$ and then $B$, is there a single matrix that does both in one go?
3. And the one that decides this entire course: **if I chain many matrices together, do I get something genuinely new — or just another matrix?**

---

## 2. What Would an Answer Need?

To describe "what a matrix does to space" usefully, we need a description that:

1. **Covers every point** — there are infinitely many, so we cannot list them.
2. **Is finite.** Four numbers describe a $2\times2$ matrix; the description must be that small.
3. **Predicts.** Given any $\mathbf{x}$, it should tell us where $\mathbf{x}$ lands without re-deriving anything.
4. **Composes.** It must answer question 2 above — what "do $A$, then $B$" is.

---

## 3. First Attempt: Track Every Point

The direct approach: apply $A$ to lots of points and tabulate where they go.

$$
(1,1) \to (2,3), \quad (2,0) \to (4,0), \quad (0,5) \to (0,15), \quad (3,7) \to (6,21), \ldots
$$

It is correct, and it is useless. The plane has infinitely many points, so the table never ends — and looking at it tells you nothing about *why* those are the destinations. This is Chapter 3 §3's failure wearing new clothes: data without structure.

> ⚠️ **A Tempting Wrong Idea**
>
> *"A transformation is just a big lookup table of where points go."*
>
> If that were true, four numbers could not possibly specify it. The fact that a $2\times2$ matrix — **four numbers** — pins down the fate of infinitely many points means the movement has enormous structure. Find the structure and you never need the table.

---

## 4. The Discovery: Follow Only the Basis

Here is the trick that unlocks everything. Do not track arbitrary points. Track the two simplest ones:

$$
\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
\qquad
\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

These are the **basis vectors** — one step east, one step north. Every other point is built from them. The point $(3, 7)$ is literally $3\mathbf{e}_1 + 7\mathbf{e}_2$: walk 3 east, then 7 north.

Now watch what a matrix does to them:

$$
A\mathbf{e}_1 = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}
\qquad
A\mathbf{e}_2 = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 3 \end{bmatrix}
$$

Look at the answers, then look back at $A$. $A\mathbf{e}_1$ is the **first column**. $A\mathbf{e}_2$ is the **second column**. That is not a coincidence of this example — multiplying by $\mathbf{e}_1$ selects column 1 by construction, because it multiplies the first column by 1 and every other column by 0.

$$
\boxed{\;\text{The columns of a matrix are where the basis vectors land.}\;}
$$

That single sentence is the whole chapter. A matrix is not a grid of numbers; it is **a record of where east and north end up.**

| Level | The same idea |
|---|---|
| 💡 **Intuition** | Space is a sheet of graph paper. A matrix tells you where to pin the two arrows that define the grid. Everything else on the paper follows along, stretching with it. |
| ✏️ **Numbers** | $A = \begin{bmatrix} 2 & 0 \\ 0 & 3\end{bmatrix}$ says: east goes to $(2,0)$, north goes to $(0,3)$. So the grid stretches 2× wide and 3× tall. |
| 🎓 **Abstraction** | $A\mathbf{e}_j = \mathbf{a}_j$, the $j$-th column. The map is determined by the images of a basis. |

> 📜 **History Lens — Cayley's Real Motivation, and Klein's Big Idea**
>
> Chapter 3 promised the other half of this story. When Arthur Cayley defined matrix multiplication in his 1858 *Memoir*, he was not thinking about datasets. He was substituting one change of variables into another — if $x' = ax + by$ describes one transformation and you feed its output into a second, what single set of coefficients describes the combined effect? The multiplication rule was **defined so that it would mean composition**, which is why §9 below recovers exactly the formula Chapter 3 derived from pricing houses in parallel.
>
> Fourteen years later, Felix Klein gave his 1872 lecture at Erlangen proposing something audacious: that a geometry *is* the study of the properties preserved by a group of transformations. Euclidean geometry is what survives rotation and translation; other geometries are what survive other transformation groups. Geometry stopped being about shapes and became about **what stays the same when things move**.
>
> That idea is not a historical curiosity for us. Chapter 38 is built on it — a convolutional network is, precisely, a network designed around which transformations should leave the answer unchanged.

---

## 5. Why the Basis Is Enough

Let us prove that tracking two arrows really does determine everything. Take any point $\mathbf{x} = (x_1, x_2)$ and write it in terms of the basis:

$$
\mathbf{x} = x_1\mathbf{e}_1 + x_2\mathbf{e}_2
$$

Matrix multiplication distributes over addition and pulls out scalars — check it against the entry formula from Chapter 3 §8 if you want to see why. So:

$$
A\mathbf{x} = A(x_1\mathbf{e}_1 + x_2\mathbf{e}_2) = x_1(A\mathbf{e}_1) + x_2(A\mathbf{e}_2) = x_1\mathbf{a}_1 + x_2\mathbf{a}_2
$$

Read that last expression carefully. **The output is built from the transformed basis vectors using the original coordinates.** The numbers $x_1, x_2$ never change; what changes is what they are counting.

Those two properties — that $A(\mathbf{u}+\mathbf{v}) = A\mathbf{u} + A\mathbf{v}$ and $A(c\mathbf{u}) = c\,A\mathbf{u}$ — are the definition of a **linear map**. Geometrically they say something concrete:

- grid lines stay straight and parallel;
- the origin does not move;
- evenly spaced points stay evenly spaced.

A transformation that bends lines, or moves the origin, is not linear and no matrix can express it. That limitation is not a footnote — it is the reason Chapter 25 exists.

---

## 6. The Four Things a Matrix Can Do

With "columns are where the basis lands," you can now read a matrix by eye. Take the unit square with corners $(0,0), (1,0), (1,1), (0,1)$ and watch.

**Scale** — stretch each axis:

$$
\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}:
\quad \mathbf{e}_1 \to (2,0), \quad \mathbf{e}_2 \to (0,3)
$$

The square becomes a $2\times3$ rectangle.

**Rotate** — turn everything 90° anticlockwise:

$$
\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}:
\quad \mathbf{e}_1 \to (0,1), \quad \mathbf{e}_2 \to (-1,0)
$$

East becomes north, north becomes west. The square is unchanged in size — just turned.

**Shear** — push the top sideways while the bottom stays put:

$$
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}:
\quad \mathbf{e}_1 \to (1,0), \quad \mathbf{e}_2 \to (1,1)
$$

The square becomes a leaning parallelogram. Note $\mathbf{e}_1$ did not move at all — **hold that thought until §12.**

**Project** — flatten everything onto the $x$-axis:

$$
\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}:
\quad \mathbf{e}_1 \to (1,0), \quad \mathbf{e}_2 \to (0,0)
$$

North collapses to nothing. The square becomes a line segment. This one is different in kind from the other three, and §8 explains why.

---

## 7. The Determinant: How Much Does Area Change?

Each transformation did something to the unit square's area. The square starts with area 1. After the transformation it is a parallelogram with some other area, and the ratio is a property of the matrix alone.

That ratio is the **determinant**. For a $2\times2$ matrix:

$$
\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

Check it against §6:

| Transformation | Matrix | $\det$ | Area of the square afterwards |
|---|---|---:|---|
| scale | $\begin{bmatrix} 2&0\\0&3\end{bmatrix}$ | $6$ | $2 \times 3 = 6$ ✓ |
| rotate | $\begin{bmatrix} 0&-1\\1&0\end{bmatrix}$ | $1$ | turning changes nothing ✓ |
| shear | $\begin{bmatrix} 1&1\\0&1\end{bmatrix}$ | $1$ | same base, same height ✓ |
| project | $\begin{bmatrix} 1&0\\0&0\end{bmatrix}$ | $0$ | flattened to a line — area 0 ✓ |

> 💡 **Intuition** — $\det A$ is the factor by which the transformation multiplies areas (volumes in 3-D, and the same idea in any dimension). A negative determinant means space was flipped over, like turning the paper face-down.

And $\det A = 0$ is the alarm bell. It means space was **squashed into something lower-dimensional** — and that is not recoverable.

---

## 8. When a Matrix Destroys Information

Look again at the projection $P = \begin{bmatrix} 1&0\\0&0\end{bmatrix}$ and ask where the point $(5, 9)$ goes. It goes to $(5, 0)$. Where does $(5, 100)$ go? Also $(5,0)$. And $(5, -3)$? $(5,0)$.

An entire infinite line of points collapses onto one point. So if I hand you the output $(5,0)$ and ask what the input was, **you cannot tell me.** The information is gone, not hidden.

Two names make this precise:

- The **null space** (or kernel) is the set of vectors sent to the origin. For $P$, that is the whole $y$-axis — a 1-dimensional space of inputs that all vanish.
- The **rank** is the dimension of what survives — the number of independent directions in the output. $P$ has rank 1: everything lands on a line.

These are tied together by one of linear algebra's cleanest facts, the **rank–nullity theorem**:

$$
\text{rank} + \text{nullity} = \text{number of input dimensions}
$$

For $P$: $1 + 1 = 2$ ✓. Every dimension is either preserved or destroyed; none goes missing unaccounted for.

> ⚠️ **A note on the word "rank."** In Chapter 3 rank meant *how many axes a tensor has* — a $(4,2)$ matrix has rank 2 in that sense, always. Here rank means *how many independent directions survive the transformation*, which for a $2\times2$ matrix can be 2, 1 or 0. The same word, two unrelated meanings, and only context to tell them apart.

A matrix with $\det = 0$ has rank less than full, has a non-trivial null space, and **has no inverse** — all three statements say the identical thing: information was destroyed. You will meet this the first time a linear system refuses to solve, and Chapter 13 will meet it again as multicollinearity.

---

## 9. The Discovery: Composition *Is* Matrix Multiplication

Now question 2 from §1. Apply $B$ first, then $A$. Is there a single matrix that does both?

Use §4's tool: a transformation is determined by where the basis vectors land, so just follow $\mathbf{e}_j$ through both steps.

**Step 1.** $B\mathbf{e}_j = \mathbf{b}_j$, the $j$-th column of $B$.

**Step 2.** Now apply $A$ to that. By §5, $A$ acting on the vector $\mathbf{b}_j$ gives

$$
A\mathbf{b}_j = \sum_k (\mathbf{b}_j)_k \,\mathbf{a}_k = \sum_k B_{kj}\,\mathbf{a}_k
$$

Take entry $i$ of that result:

$$
\big(A(B\mathbf{e}_j)\big)_i = \sum_k A_{ik} B_{kj}
$$

Compare it with Chapter 3 §8, where the same expression came out of pricing four houses with three weight vectors:

$$
(AB)_{ij} = \sum_{k} A_{ik}B_{kj}
$$

**Identical.** So the matrix whose columns are "where the basis lands after doing $B$ then $A$" is exactly the product $AB$:

$$
\boxed{\;A(B\mathbf{x}) = (AB)\mathbf{x}\;}
$$

This is Cayley's route, arriving at Chapter 3's rule from the opposite direction. Two completely unrelated problems — parallel bookkeeping and composing movements — produce the same operation. That is the strongest evidence there is that matrix multiplication is a real thing about the world rather than a convention someone chose.

Note the order: $AB$ means **do $B$ first**. It reads right to left, like $f(g(x))$.

---

## 10. Why Order Matters, Geometrically

Chapter 3 §9 showed $AB \ne BA$ with two small matrices and left the meaning for later. Later is now.

Take the shear $S$ and the rotation $R$ from §6:

$$
SR = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}
\qquad
RS = \begin{bmatrix} 0 & -1 \\ 1 & 1 \end{bmatrix}
$$

Different matrices, so different transformations. And once matrices are *actions*, that stops being a strange algebraic defect and becomes obvious:

> Put on socks, then shoes. Or put on shoes, then socks. Same two actions, wildly different outcomes.

Rotating a leaning shape is not the same as leaning a rotated shape. The notebook draws both so you can see the two results side by side.

---

## 11. 🔬 The Experiment: Does Depth Buy Anything?

Now the question this whole chapter was built to answer, and the one that shapes every neural network in this course.

A network stacks layers. Each layer, so far, is a matrix multiply. So a three-layer network computes

$$
\mathbf{y} = W_3\big(W_2(W_1\mathbf{x})\big)
$$

> 🧠 **Predict before reading on.** Three matrices, applied in sequence. Is the overall transformation more expressive than one matrix — can it do something no single matrix can?

By §9, associativity lets us regroup:

$$
W_3\big(W_2(W_1\mathbf{x})\big) = \big(W_3W_2W_1\big)\mathbf{x} = C\mathbf{x}
$$

Multiply the three matrices together first, and $C$ is just a matrix. Concretely, with

$$
W_1 = \begin{bmatrix} 1&2\\0&1 \end{bmatrix},
W_2 = \begin{bmatrix} 2&0\\1&1 \end{bmatrix},
W_3 = \begin{bmatrix} 0&1\\-1&0 \end{bmatrix}
\qquad\Longrightarrow\qquad
C = \begin{bmatrix} 1&3\\-2&-4 \end{bmatrix}
$$

Feed $(3,2)$ through all three one at a time, and feed it through $C$ once. Both give $(9,-14)$. Not approximately — exactly, and for every input.

> ⚠️ **This is a catastrophe for deep learning, and it is worth feeling the weight of it.**
>
> A hundred stacked linear layers — a million parameters — collapse into a single matrix with four numbers. The depth is *entirely fictitious*. Whatever a deep linear network can compute, one linear layer can compute, and it can only ever draw straight boundaries.

So the answer to §1's third question is **no**. Chaining matrices gains nothing.

This is not a flaw in our reasoning; it is a true fact about linear maps, and it means something must be added between the layers — something that is *not* linear. That is precisely the job of the activation function, and precisely why Chapters 24 and 25 exist.

---

## 12. What Survives?

One more observation, and it is the seed of the next chapter.

Watch the shear $S = \begin{bmatrix} 1&1\\0&1\end{bmatrix}$ act on various vectors. Most of them get tilted — $(0,1)$ becomes $(1,1)$, a 45° swing. But look at $(1,0)$:

$$
S\begin{bmatrix} 1 \\ 0\end{bmatrix} = \begin{bmatrix} 1 \\ 0\end{bmatrix}
$$

Unmoved. And the scale matrix $\begin{bmatrix} 2&0\\0&3\end{bmatrix}$ leaves both axes pointing exactly where they started — it only changes their lengths, by 2 and by 3.

Meanwhile a 90° rotation leaves *no* direction pointing where it started. Obviously — it rotates everything.

So transformations differ in a way our four descriptions have not captured: **some directions are special to a given matrix.** They come out parallel to how they went in, merely stretched. Finding them turns out to be one of the most useful things you can do to a matrix.

---

## 13. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **Determinant zero** | the system will not solve; the inverse does not exist | Space was flattened; the information is gone. §8 |
| **Near-zero determinant** | wildly unstable answers | Almost flattened — tiny input changes cause huge output changes. This is Chapter 2's conditioning problem, seen geometrically. |
| **Assuming $AB = BA$** | subtly wrong pipelines | Order of operations is order of actions. §10 |
| **Stacking linear layers** | a million parameters that fit a straight line | §11 |
| **Expecting a matrix to bend things** | a model that cannot fit curves | Linear maps keep lines straight and fix the origin. §5 |

---

## 14. 🎯 Machine Learning Connection

| This chapter | In a network |
|---|---|
| the matrix as a movement of space | every layer relocates its input into a new space |
| columns = where the basis lands | each column of $W$ is one neuron's contribution direction |
| rank | how many independent features a layer can actually express |
| null space | input directions a layer is completely blind to |
| $\det = 0$, information destroyed | why some layers lose signal irreversibly |
| composition collapses | why activation functions are mandatory (Ch 25) |
| what a transformation preserves | the whole idea behind convolution (Ch 38) |

The useful mental model from here on: **a deep network is a sequence of spaces.** The input arrives in pixel space or word space, and each layer moves it somewhere new, hoping to land in a space where the answer is easy to read off — ideally one where a straight line suffices. Chapter 46 makes this picture precise.

---

## 15. Distinctions That Matter

| | |
|---|---|
| **Matrix as storage** — a table (Ch 3) | **Matrix as action** — a movement of space (Ch 4) |
| **Columns** — where the basis vectors land | **Rows** — how each output coordinate is assembled |
| **Rank of a tensor** — number of axes (Ch 3) | **Rank of a matrix** — surviving dimensions |
| $\det \ne 0$ — invertible, nothing lost | $\det = 0$ — collapsed, unrecoverable |
| $AB$ — do $B$, then $A$ | $BA$ — a different transformation |
| **Linear** — lines stay straight, origin fixed | **Affine** — linear, then a shift ($+\,b$, as in our bias) |

---

## 16. What We Discovered

1. A matrix does not store points; it relocates every point in space at once.
2. Tracking individual points is hopeless, but tracking the basis vectors is enough — and their destinations are exactly the columns.
3. Linearity is what makes that work: output coordinates stay the same while the things they count change.
4. The four basic movements — scale, rotate, shear, project — are readable straight off a matrix's columns.
5. The determinant is the area-scaling factor, and zero means space was flattened.
6. Flattening destroys information permanently; rank and null space measure exactly how much.
7. Composing transformations *is* matrix multiplication — the same formula Chapter 3 derived from a completely different problem.
8. $AB \ne BA$ because doing things in the other order genuinely gives a different result.
9. **Chained linear maps collapse into one linear map, so depth alone buys nothing at all.**
10. Some directions survive a transformation unchanged in direction — and those deserve their own chapter.

## 17. Mathematics We Built

$$
A\mathbf{e}_j = \mathbf{a}_j \quad (\text{column } j)
\qquad
A\mathbf{x} = x_1\mathbf{a}_1 + x_2\mathbf{a}_2
$$

$$
A(\mathbf{u}+\mathbf{v}) = A\mathbf{u}+A\mathbf{v}, \qquad A(c\mathbf{u}) = c\,A\mathbf{u}
$$

$$
\det\begin{bmatrix} a&b\\c&d\end{bmatrix} = ad-bc
\qquad
\text{rank} + \text{nullity} = n
$$

$$
A(B\mathbf{x}) = (AB)\mathbf{x}
\qquad
W_3W_2W_1 = C \;\;(\text{a single matrix})
$$

## 18. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $\mathbf{e}_1, \mathbf{e}_2$ | the basis: one step east, one step north | `np.eye(2)` |
| $\mathbf{a}_j$ | column $j$ of $A$ — where $\mathbf{e}_j$ lands | `A[:, j]` |
| $A\mathbf{x}$ | where the transformation sends $\mathbf{x}$ | `A @ x` |
| $\det A$ | area-scaling factor | `np.linalg.det(A)` |
| rank | surviving dimensions | `np.linalg.matrix_rank(A)` |
| null space | inputs sent to the origin | `scipy.linalg.null_space(A)` |
| $AB$ | do $B$ first, then $A$ | `A @ B` |
| $A^{-1}$ | the undo, when $\det A \ne 0$ | `np.linalg.inv(A)` |

## 19. One-Minute Explanation

With no equations:

> A matrix is four numbers. How can four numbers decide where *every* point in an infinite plane ends up — and why does stacking a hundred of them achieve nothing?

---

## 20. Exercises

**Level 1 — Observe.** Look at the four matrices in §6 and, without calculating, say which ones change the area of the unit square and which do not. Then look at $\begin{bmatrix} 0&1\\1&0\end{bmatrix}$: where do $\mathbf{e}_1$ and $\mathbf{e}_2$ land, what shape does the square become, and what is unusual about its determinant?

**Level 2 — Calculate (by hand).** For $A = \begin{bmatrix} 3&1\\1&2\end{bmatrix}$: find where $\mathbf{e}_1$ and $\mathbf{e}_2$ land, compute $\det A$, and sketch the image of the unit square. Then compute $A\begin{bmatrix}2\\1\end{bmatrix}$ two ways — directly, and as $2\mathbf{a}_1 + 1\mathbf{a}_2$ using §5 — and confirm they agree.

**Level 3 — Derive.** Prove that $\det(AB) = \det(A)\det(B)$ for $2\times2$ matrices by multiplying out both sides. Then explain what it *means* geometrically in one sentence. Finally, use it to prove that if $\det A = 0$ then $AB$ can never be invertible, no matter what $B$ is.

**Level 4 — Investigate** (notebook Steps 8–11). Build a chain of $n$ random $2\times2$ matrices for $n = 2, 5, 20, 100$ and verify the composite equals a single matrix every time. Then measure the determinant of the composite as $n$ grows. What happens to it, and what does that predict about very deep linear networks before we have even added activations?

**Level 5 — Design.** You need a transformation that moves the unit square one step to the right — every point $(x,y) \to (x+1, y)$. Prove no $2\times2$ matrix can do this. (Hint: where must the origin go, and where does §5 say the origin must go?) Then design a fix: find a $3\times3$ matrix that achieves it on points written as $(x, y, 1)$. What did you have to give up, and where have you already seen this trick? (Look at the $+b$ in our model.)

---

## 21. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "The rows of $A$ are where the basis vectors go." | The **columns** are. $A\mathbf{e}_1$ picks out column 1. §4 |
| "$AB$ means do $A$ first." | It means do **$B$** first. It reads right to left, like $f(g(x))$. §9 |
| "A determinant is just a number you compute for exams." | It is the area-scaling factor, and zero is a statement that information was destroyed. §7–8 |
| "Deep linear networks are more powerful than shallow ones." | They are provably identical. §11 |
| "$\det = 0$ means the matrix is all zeros." | It means space was flattened. $\begin{bmatrix}1&0\\0&0\end{bmatrix}$ has plenty of non-zero entries. |
| "A matrix can move the origin." | Never. $A\mathbf{0} = \mathbf{0}$ always — that is why the bias $b$ exists separately. |

## 22. Socratic Questions

1. Every linear map fixes the origin. Our model has always been $\mathbf{w}\cdot\mathbf{x} + b$ — so is the model linear? What exactly is the $+b$ doing that a matrix cannot?
2. The determinant of a rotation is 1, and of a projection 0. What kind of transformation would have a *negative* determinant, and what does it do to a shape?
3. If $\det A$ is very small but not zero, $A$ is technically invertible. Should you trust the inverse? (Chapter 2 §10 already told you the answer in different language.)
4. A layer with rank 1 sends everything onto a line. What would that mean for a network trying to distinguish ten classes?
5. §11 shows depth is fictitious for linear maps. Does that mean depth is *always* pointless, or only under a specific condition — and what exactly is that condition?
6. We said the columns tell you everything about a transformation. So what do the **rows** tell you? Is there a question they answer better?

---

## 23. 🔭 Bridge to Chapter 05

We can now read a matrix as an action: the columns say where the grid goes, the determinant says how area changes, the rank says how much survives, and multiplication says how actions compose.

But §12 left something unresolved. Most vectors get twisted by a transformation — they come out pointing somewhere new. A few do not. The shear left the $x$-axis exactly where it was; the scaling matrix left both axes pointing the same way and merely changed their lengths; the rotation left nothing at all.

Those surviving directions are clearly special. They are the axes the transformation is "really" built around — the skeleton it acts along. And if we could find them for any matrix, we could understand that matrix as *nothing more than stretching along a few fixed directions*, which is about as simple as a transformation could possibly be.

> **Which directions does a matrix leave pointing where they started, by how much does it stretch them, and what can we do once we know?**

That is where Chapter 05 begins — and it ends with a tool that compresses images, finds the structure in data, and explains what a neural network layer is really doing to its input.

➡️ **Next:** [Chapter 05 — Eigenvectors: What a Transformation Leaves Alone](<../Lecture 05 - Eigenvectors: What a Transformation Leaves Alone/blog.md>)
