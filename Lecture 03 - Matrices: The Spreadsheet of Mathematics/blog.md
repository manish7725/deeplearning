# Lecture 03 — Matrices: The Spreadsheet of Mathematics

> **The Big Question:** How can we hold many vectors as a single object, and compute with all of them in one operation?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2003%20-%20Matrices%3A%20The%20Spreadsheet%20of%20Mathematics/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** One house became one vector $\mathbf{x}$, and one dot product $\mathbf{w}\cdot\mathbf{x} + b$ priced it. We also learned that feature scale controls the shape of the loss landscape.
**Today:** One house at a time is not enough. We build the object that prices an entire dataset — and every valuation of it — in a single operation.
**Next:** We will notice that our new object did not just *store* numbers, it *moved* them. That question takes us into the geometry of **transformations**.

---

## 1. The Problem: Four Houses, Four Dot Products, and a Million More

The office has the same four houses:

| House | Rooms | Area (sq ft) | Price (₹ lakh) |
|---|---:|---:|---:|
| A | 2 | 800 | 9 |
| B | 2 | 1200 | 11 |
| C | 3 | 900 | 11.5 |
| D | 4 | 1600 | 17 |

Chapter 2 gave us the rule $\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$ with $\mathbf{w} = [2,\, 0.005]$ and $b = 1$. To price all four, we run it four times:

$$
\hat{y}_A = \mathbf{w}\cdot\mathbf{x}_A + b = 9
\qquad
\hat{y}_B = \mathbf{w}\cdot\mathbf{x}_B + b = 11
$$
$$
\hat{y}_C = \mathbf{w}\cdot\mathbf{x}_C + b = 11.5
\qquad
\hat{y}_D = \mathbf{w}\cdot\mathbf{x}_D + b = 17
$$

Four correct prices. So what is the problem?

Write down the mathematics for a million houses. Not the code — the *mathematics*. You cannot. There is no expression for "do this dot product for every row" in the notation we currently own; there is only a list that grows as long as the dataset. Every gradient we derive, every proof we attempt, would have to be written a million times.

And Chapter 1 warned us this matters. We are going to differentiate these expressions. A notation you cannot write down is a notation you cannot do calculus on.

> 🧠 **Think** — This is the same wall as Chapter 2 §3, one level up. There, writing $w_1x_1 + w_2x_2 + \cdots$ for a million *features* was impossible, and the vector fixed it. Now writing one equation per *house* is impossible. **We do not need a new idea. We need the same idea applied to a new axis.**

---

## 2. What Would a Solution Need?

1. **Hold many vectors as one object**, with a single name.
2. **Keep every vector's identity.** House C must remain recoverable; the houses must not blend.
3. **Produce all the answers in one operation**, whose written form does not grow with the dataset.
4. **Reduce to Chapter 2** when there is only one house. A new notation that contradicts the old one is a new bug.
5. **Be checkable.** With a million numbers in flight, we need a way to know an expression is wrong *before* running it.

Requirement 5 is unusual and turns out to be the most practically valuable thing in this chapter.

---

## 3. First Attempt: Keep a List of Vectors

We already have vectors. Why not just keep four of them in a list?

$$
\text{data} = \big[\; \mathbf{x}_A,\; \mathbf{x}_B,\; \mathbf{x}_C,\; \mathbf{x}_D \;\big]
$$

This satisfies requirements 1 and 2 — everything is stored, nothing is blended. It fails requirement 3 completely. A list is a container, not a mathematical object. You cannot write

$$
\hat{\mathbf{y}} = \text{data} \cdot \mathbf{w} + b
$$

because nothing defines what it means to "dot" a list with a vector. To extract the answers you must still loop, one house at a time, and the loop is not mathematics — it is bookkeeping wearing mathematics' clothes.

> ⚠️ **A Tempting Wrong Idea**
>
> *"A matrix is just a 2-D array — a table of numbers."*
>
> The table is the least interesting part. NumPy would happily store our four houses as a nested list, and that would give us exactly nothing. **What makes a matrix a matrix is the algebra defined on it**: a product that computes every dot product at once, a transpose, a shape rule that catches errors. Storage without operations is a spreadsheet. We need a spreadsheet that multiplies.

---

## 4. The Discovery: The Matrix

Take the four house vectors and stack them as **rows** of one rectangular object:

$$
X =
\begin{bmatrix}
2 & 800 \\
2 & 1200 \\
3 & 900 \\
4 & 1600
\end{bmatrix}
$$

One symbol, $X$, now holds the entire dataset. Read it like a spreadsheet: **each row is one house, each column is one feature.**

| Level | The same idea |
|---|---|
| 💡 **Intuition** | A filing cabinet where every drawer holds one house's backpack, and the drawers are lined up so you can reach into the same pocket of all of them at once. |
| ✏️ **Numbers** | Row 3 is house C: 3 rooms, 900 sq ft. Column 2 is *every* house's area: $[800, 1200, 900, 1600]$. |
| 🎓 **Abstraction** | $X \in \mathbb{R}^{n \times d}$ — $n$ rows (examples), $d$ columns (features). Here $n = 4$, $d = 2$. |

The notation for a single entry is $X_{ij}$: row $i$, column $j$, **always in that order**. $X_{32} = 900$ is house C's area. Rows before columns is a convention you will repeat ten thousand times; learn it now.

> 📜 **History Lens — Cayley, Sylvester, and 2,000 Years of Arrays**
>
> Rectangular arrays of numbers are ancient. The Chinese *Nine Chapters on the Mathematical Art*, compiled around two thousand years ago, solves systems of linear equations by writing the coefficients in a rectangular array and systematically eliminating entries — essentially the method Gauss would publish in Europe in the early 1800s.
>
> But for all that time the array was a **worksheet**, not an object. You did things *to* it; you could not do arithmetic *with* it.
>
> That changed in the 1850s, in a friendship between two London mathematicians. James Joseph Sylvester coined the word **matrix** in 1850 — Latin for *womb*, the thing from which determinants are born. Then Arthur Cayley, in *A Memoir on the Theory of Matrices* (1858), did something genuinely new: he treated the array as a single algebraic quantity that could be added and multiplied.
>
> And here is the part worth remembering. Cayley did not invent his multiplication rule for the reason we are about to invent it. He arrived at it from **composing transformations** — substitute one change of variables into another, and ask what single array describes the combined effect. The rule that falls out is exactly the one we are about to derive from a completely different problem, pricing houses in parallel.
>
> Two unrelated questions, one operation. That is usually a sign you have found something real — and Chapter 4 is the other half of the story.

---

## 5. The Discovery: Matrix × Vector

Now define the operation the problem actually asked for. We want every house's price at once, and we know each price is one dot product between $\mathbf{w}$ and a row. So define the product $X\mathbf{w}$ to be exactly that — **one dot product per row**, results stacked in order:

$$
X\mathbf{w} =
\begin{bmatrix}
\text{row}_1 \cdot \mathbf{w} \\
\text{row}_2 \cdot \mathbf{w} \\
\text{row}_3 \cdot \mathbf{w} \\
\text{row}_4 \cdot \mathbf{w}
\end{bmatrix}
$$

Nothing has been invented. We named a pattern we were already performing. With $\mathbf{w} = [2,\, 0.005]$:

$$
X\mathbf{w} =
\begin{bmatrix}
(2)(2) + (800)(0.005) \\
(2)(2) + (1200)(0.005) \\
(3)(2) + (900)(0.005) \\
(4)(2) + (1600)(0.005)
\end{bmatrix}
=
\begin{bmatrix}
4 + 4 \\ 4 + 6 \\ 6 + 4.5 \\ 8 + 8
\end{bmatrix}
=
\begin{bmatrix}
8 \\ 10 \\ 10.5 \\ 16
\end{bmatrix}
$$

Add $b = 1$ to every entry and the entire dataset is priced:

$$
\boxed{\;\hat{\mathbf{y}} = X\mathbf{w} + b\;}
=
\begin{bmatrix} 9 \\ 11 \\ 11.5 \\ 17 \end{bmatrix}
\;\checkmark
$$

One line. It is the same line for four houses or four million — only $n$ changes, and $n$ does not appear in the formula. Requirement 3, satisfied.

And requirement 4 holds too: with a single house, $X$ has one row, and $X\mathbf{w}$ is a single dot product — Chapter 2 exactly, unchanged.

---

## 6. The Shape Rule

Look closely at what made that work. Each output entry was a dot product between a **row of $X$** (length 2) and $\mathbf{w}$ (length 2). A dot product pairs up slots, so those two lengths *must be equal* — otherwise there is a slot with no partner and the operation is undefined.

That single observation is the whole shape rule:

$$
X \in \mathbb{R}^{n \times d}, \quad \mathbf{w} \in \mathbb{R}^{d}
\qquad \Longrightarrow \qquad
X\mathbf{w} \in \mathbb{R}^{n}
$$

Written as shapes, with the inner dimensions meeting in the middle:

```text
   (4 × 2) · (2,)  ->  (4,)
        ↑     ↑
        └─────┘
     these must match — they are what gets summed over,
     and they vanish from the result
```

The dimension that matches is **consumed**. The dimensions that survive — how many rows, and whatever the other operand contributes — become the output shape. Everything else in this chapter follows from that sentence.

---

## 7. The Problem Again: Three Valuations at Once

The office comes back. They no longer want one number per house; they want three:

- **market price** — what it should sell for;
- **insurance value** — the cost to rebuild, which depends on area and barely on room count;
- **tax assessment** — which the municipality sets at 60% of market value.

Each valuation is its own question, so each has its own weight vector:

$$
\mathbf{w}_{\text{market}} = \begin{bmatrix} 2 \\ 0.005 \end{bmatrix}
\quad
\mathbf{w}_{\text{ins}} = \begin{bmatrix} 0 \\ 0.004 \end{bmatrix}
\quad
\mathbf{w}_{\text{tax}} = \begin{bmatrix} 1.2 \\ 0.003 \end{bmatrix}
$$

with biases $1$, $0$ and $0.6$. Note $\mathbf{w}_{\text{ins}}$ has a zero in the rooms slot — for rebuild cost, the number of rooms genuinely does not matter. A weight of zero is the model saying *"ignore this."*

Now count the work: 4 houses × 3 valuations = **12 dot products**. We are back where §1 started, one axis over. Chapter 2 stacked measurements; §4 stacked houses; the thing left unstacked is the set of weight vectors.

So stack them — as **columns**, since each one produces a column of answers:

$$
W =
\begin{bmatrix}
2 & 0 & 1.2 \\
0.005 & 0.004 & 0.003
\end{bmatrix}
\in \mathbb{R}^{2 \times 3}
$$

---

## 8. The Discovery: Matrix × Matrix

We want the entry in row $i$, column $j$ of the answer to be *house $i$ valued by method $j$* — which is row $i$ of $X$ dotted with column $j$ of $W$. Write that down and the definition of matrix multiplication appears:

$$
(XW)_{ij} = \sum_{k=1}^{d} X_{ik} W_{kj}
$$

Read the sigma aloud: *"walk along row $i$ of $X$ and down column $j$ of $W$ together, multiply each meeting pair, and total them."* The index $k$ is the feature axis — the thing being summed over, and the thing that disappears.

Do one entry by hand. Row 1 (house A) and column 3 (tax):

$$
(XW)_{13} = (2)(1.2) + (800)(0.003) = 2.4 + 2.4 = 4.8
$$

add the tax bias $0.6$, and house A's assessment is ₹5.4 lakh. Check it against the rule the municipality actually uses: 60% of ₹9 lakh market value $= 5.4$ ✓.

All twelve numbers, from one operation:

$$
\hat{Y} = XW + \mathbf{b} =
\begin{bmatrix}
9 & 3.2 & 5.4 \\
11 & 4.8 & 6.6 \\
11.5 & 3.6 & 6.9 \\
17 & 6.4 & 10.2
\end{bmatrix}
$$

Every row is a house; every column is a valuation. The shape rule does its work exactly as before:

```text
   (4 × 2) · (2 × 3)  ->  (4 × 3)
        ↑     ↑
        └─────┘  consumed
```

> 💡 **Intuition** — Matrix multiplication is **every question asked of every example, all at once**. Rows carry the things you have; columns carry the questions you want answered; each cell is one answer.

---

## 9. Why *This* Definition?

The rule looks arbitrary the first time you meet it. Why sum over that axis? Why not just multiply matching cells, the way we add matching cells?

Element-wise multiplication is a perfectly real operation — it has a name, the **Hadamard product**, written $A \odot B$. Try it here and watch it fail:

$X$ is $4\times2$ and $W$ is $2\times3$. They are not even the same shape, so $X \odot W$ **does not exist**. Force it by picking two matrices of equal shape and it computes something we never asked for: a grid of isolated products, with no summing, therefore no dot products, therefore no prices.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Multiplication of matrices should work like addition of matrices — cell by cell."*
>
> Addition *is* cell by cell, and that is exactly why addition cannot price a house. Pricing requires **combining** several measurements into one number. Combining means summing across an axis, and the definition in §8 is simply the bookkeeping for "which axis gets summed." Matrix multiplication is not element-wise because **the problem is not element-wise.**

Two consequences follow immediately, both surprising the first time.

**Order matters.** $AB$ and $BA$ are different operations, and usually different numbers:

$$
A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
B = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}
\qquad
AB = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}, \quad
BA = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}
$$

Same two matrices, different answers. This is not a defect — Chapter 4 will show that $AB \ne BA$ says something true about the world: doing two things in the other order genuinely gives a different result.

**Most pairs cannot be multiplied at all.** $W X$ would need $(2\times3)\cdot(4\times2)$, and $3 \ne 4$, so it is undefined. Roughly speaking, shapes refuse far more often than they agree — which is what makes them such a good error detector.

---

## 10. Transpose: Summing Over Houses Instead of Features

Here is a question our notation cannot yet answer. In Chapter 2's training loop the gradient line read

```python
w -= learning_rate * (2 / n) * (X.T @ error)
```

Why the `.T`?

Think about what the gradient must be. Chapter 1 §11 says $\frac{\partial L}{\partial w_j} = \frac{1}{n}\sum_i 2(\hat{y}_i - y_i)\,x_{ij}$ — for each **feature** $j$, sum over all **houses** $i$. But $X$ is organized the other way: its rows are houses. Multiplying $X$ by something sums over *features*, because features are the inner axis.

We need to sum over the other axis. So define the **transpose**: flip the matrix across its diagonal, turning rows into columns.

$$
X = \begin{bmatrix} 2 & 800 \\ 2 & 1200 \\ 3 & 900 \\ 4 & 1600 \end{bmatrix} \in \mathbb{R}^{4\times2}
\qquad
X^{T} = \begin{bmatrix} 2 & 2 & 3 & 4 \\ 800 & 1200 & 900 & 1600 \end{bmatrix} \in \mathbb{R}^{2\times4}
$$

Formally $(X^T)_{ij} = X_{ji}$ — the entry at row $i$, column $j$ of the transpose is the entry at row $j$, column $i$ of the original. Now each *row* of $X^T$ is one feature across all houses, so multiplying by $X^T$ sums over houses. Exactly what the gradient needed.

---

## 11. Shapes Tell You the Formula

Now the payoff, and the habit this chapter exists to build.

We have $X$ of shape $(4, 2)$ and an error vector of shape $(4,)$, and we need a gradient with the **same shape as $\mathbf{w}$**, which is $(2,)$ — because the update $\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla$ subtracts it from $\mathbf{w}$ directly.

Ask what combinations are even legal:

| Attempt | Shapes | Result |
|---|---|---|
| `X @ error` | $(4,2) \cdot (4,)$ | ❌ inner dims $2 \ne 4$ — undefined |
| `error @ X` | $(4,) \cdot (4,2)$ | ✅ gives $(2,)$ |
| `X.T @ error` | $(2,4) \cdot (4,)$ | ✅ gives $(2,)$ |

Two survive, and they compute the same sum. One is ruled out before a single number is touched.

> 💡 **This is requirement 5, delivered.** You did not need to know calculus to reject `X @ error` — you needed to know that 2 and 4 are different. In a model with a hundred layers, shape reasoning is the error detector that runs in your head, and it catches most mistakes before the code does.

The ritual, worth doing out loud every time you write a matrix expression:

```text
1. What shape is each thing?
2. Which axis is being summed over — is it the one I mean?
3. What shape must come out?
4. Do the inner dimensions actually touch?
```

---

## 12. Beyond Two Axes: Tensors

The office starts attaching a **photograph** to every listing, and our filing cabinet runs out of room.

Here is why. A greyscale photo is a grid of brightness values — which is already a matrix, say $64 \times 64$. So one house now carries a $(64, 64)$ object. Fine. But we have four houses, and we want all four photos in one place. We would have to stack four matrices, and a matrix has exactly two axes. There is nowhere to put them.

Add colour and it gets worse: a colour photo is three grids stacked — red, green, blue — so a single photo is already $(3, 64, 64)$.

> 🧠 **Think** — Notice the pattern this chapter keeps repeating. Every time we needed "many of something," we added an axis. Many measurements → a **vector** (Chapter 2). Many houses → a **matrix** (§4). Many photos → the next rung. There is no new idea here, only the same idea again.

So take the ladder seriously and name every rung:

| Name | Axes | Shape | Our example |
|---|---:|---|---|
| **scalar** | 0 | `()` | one price, ₹9 lakh |
| **vector** | 1 | `(2,)` | one house: $[2, 800]$ |
| **matrix** | 2 | `(4, 2)` | the dataset $X$: 4 houses × 2 features |
| **3-tensor** | 3 | `(4, 64, 64)` | 4 greyscale photos |
| **4-tensor** | 4 | `(4, 3, 64, 64)` | 4 colour photos: houses × channels × height × width |

A **tensor** is the general object: a block of numbers with any number of axes. Its **rank** is how many axes it has, and its **shape** lists how long each axis is. To pull out a single number you need one index per axis — $T_{2,1,17,40}$ is house 2, colour channel 1, row 17, column 40.

Every rung is a special case of the rung above. A matrix *is* a rank-2 tensor; a vector *is* a rank-1 tensor; a scalar *is* a rank-0 tensor. This is exactly why PyTorch calls a number, a house, a dataset and a batch of photographs by the same name: `torch.Tensor`. One object, one algebra, any number of axes.

### The shape rule survives

The discipline from §6 and §11 does not change — there is simply more to track. When a batch of examples meets a weight matrix, the **last axis is consumed** and the leading axes ride along untouched:

```text
   (32 × 100 × 64) · (64 × 10)  ->  (32 × 100 × 10)
                ↑     ↑
                └─────┘  consumed, exactly as in section 6
```

Read that as: 32 documents, each holding 100 words, each word described by 64 numbers, all pushed through a layer with 10 outputs. It is §8's operation performed 32 × 100 times and written once. That leading axis — the **batch** — appears in essentially every deep learning program you will ever read.

> ⚠️ **Two warnings about words.**
>
> A physicist or geometer means something stricter by "tensor": an object with particular transformation rules under a change of coordinates. In deep learning the word simply means **$n$-dimensional array**. Both usages are standard in their own fields; only the second is meant in this course.
>
> Worse, **rank** has two unrelated meanings. Here it means *number of axes*. In linear algebra — and in Chapter 4 — the rank of a matrix means something entirely different: how many independent directions it actually spans. Context is the only thing that tells you which is meant.

We will not develop the full mechanics now — broadcasting, reshaping, permuting axes, summing along a chosen axis. Those arrive in **Chapter 11**, at the point where convolution genuinely forces us to use them. What matters today is the ladder itself: *scalar → vector → matrix → tensor is one idea applied repeatedly, and the shape rule holds all the way up.*

---

## 13. 🔬 The Experiment: Why We Do Not Write Loops

We now have two ways to price a dataset: a Python loop over rows, and `X @ w`. They compute identical numbers. Are they equally good?

> 🧠 **Predict before running.** For 2,000 houses with 50 features each, how much faster is the matrix form than the loop — 2×, 10×, or more than 100×? And *why* would it be faster at all, given both do the same 100,000 multiply-adds?

The notebook times both (Step 8). The matrix form typically wins by two or three orders of magnitude, and the reason is worth understanding, because it explains the entire hardware story of deep learning:

- The Python loop interprets bytecode on every single multiplication, and each number is a separate boxed object scattered in memory.
- `X @ w` hands the whole array to compiled, cache-aware code that processes many numbers per instruction, in parallel, with the data laid out contiguously.

**This is the entire reason GPUs matter.** A GPU is, in effect, a machine built to do §8's operation on enormous matrices. When people say a model is "expensive to train," they mean it performs an extraordinary number of the multiply-and-sum you did by hand in §8.

> ⚠️ But note what did *not* change: the mathematics. The loop and the matrix form are the same formula. Speed is an implementation property; correctness is a mathematical one. Never let a fast implementation talk you out of checking the slow one.

---

## 14. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **Shape mismatch** | loud error, immediately | The inner dimensions disagree. This is the *good* failure — it stops you. |
| **Silent broadcasting** | no error, wrong numbers | Multiply a $(4,1)$ by a $(1,3)$ and NumPy helpfully produces a $(4,3)$ instead of complaining. The shapes were compatible; your intent was not. |
| **Wrong orientation** | plausible but wrong results | Swap rows and columns and you compute a valid expression that answers a different question. See the note below. |
| **Assuming $AB = BA$** | subtly wrong models | §9. Order encodes meaning. |
| **Summing the wrong axis** | gradients that do not train | Forgetting a `.T` sums over houses when you meant features, or vice versa. §10. |

> ⚠️ **Two conventions, and you must know which you are reading.**
>
> This course writes data as **rows**: $X$ is $(n \times d)$, and the model is $\hat{\mathbf{y}} = X\mathbf{w} + b$. Machine learning does it this way because the first axis is the *batch* — the examples you are processing together.
>
> Most linear-algebra textbooks — and Chapter 4 — write vectors as **columns**, so a transformation reads $A\mathbf{x}$, with $\mathbf{x}$ standing to the right. Both are correct; they are transposes of each other. When you meet a formula in a paper, **find out which convention it uses before trusting the shapes.** More student-hours have been lost to this than to any actual mathematics.

---

## 15. 🎯 Machine Learning Connection

The expression from §8 is not an analogy for a neural network layer. It **is** a neural network layer:

$$
\hat{Y} = XW + \mathbf{b}
$$

| Our house table | A neural network |
|---|---|
| $X$: 4 houses × 2 features | a **batch** of examples × input features |
| each column of $W$: one valuation | each column: one **neuron**, asking its own question |
| $W$: $2 \times 3$ | a layer's weights: `in_features × out_features` |
| $\hat{Y}$: 4 × 3 | the layer's output, one row per example |
| the zero in $\mathbf{w}_{\text{ins}}$ | a weight the model learned to ignore |

When you write `nn.Linear(2, 3)` in PyTorch, you are creating exactly the $W$ of §7 and the $\mathbf{b}$ beside it, and its forward pass is exactly $XW + \mathbf{b}$. A modern network is dozens of these stacked, with one extra ingredient that Chapter 6 will have to introduce for a reason Chapter 4 is about to uncover.

And the three valuations are the honest picture of what a layer's neurons are: **several different questions asked of the same input, in parallel, each with its own weights.**

---

## 16. Distinctions That Matter

| | |
|---|---|
| **Table of numbers** — storage | **Matrix** — storage *plus* an algebra: product, transpose, shape rule |
| $X\mathbf{w}$ — matrix × vector → a vector | $XW$ — matrix × matrix → a matrix |
| $XW$ — sums over the shared axis | $X \odot W$ — element-wise, sums over nothing |
| $X_{ij}$ — row $i$, column $j$ | $X_{ji}$ — the other one entirely |
| $X$ — houses in rows | $X^T$ — houses in columns; sums over the other axis |
| $AB$ | $BA$ — a different computation, often undefined |
| **Rows = examples** (this course, ML) | **Columns = vectors** (textbooks, Chapter 4) |
| **Rank of a tensor** — number of axes | **Rank of a matrix** — independent directions (Chapter 4) |
| **Matrix** — exactly 2 axes | **Tensor** — any number of axes |

---

## 17. What We Discovered

1. Writing one equation per example is as impossible as Chapter 2's one term per feature — the same wall, on a different axis.
2. A list of vectors stores the data but has no algebra, so it cannot produce the answers in one expression.
3. Stacking vectors as rows gives a matrix: one symbol for an entire dataset.
4. Matrix × vector was not invented — it is the name for "one dot product per row," which we were already doing.
5. The shape rule follows from a single fact: a dot product needs two equal-length vectors, so the inner dimensions must match and are consumed.
6. Stacking *weight* vectors as columns produces matrix × matrix, which answers every question about every example at once.
7. The definition is not arbitrary: it sums over the axis that the problem requires to be combined. Element-wise multiplication answers a question nobody asked.
8. Order matters, because $AB$ and $BA$ combine different axes.
9. Transposing switches which axis gets summed — which is why the gradient needs $X^T$.
10. Shapes can rule out a wrong formula before you compute anything, and that is the most reliable debugging tool in the course.
11. Stacking does not stop at two axes. Scalar → vector → matrix → tensor is one idea repeated, and the shape rule survives every rung.

## 18. Mathematics We Built

$$
X \in \mathbb{R}^{n \times d}
\qquad
(X^T)_{ij} = X_{ji}
\qquad
\hat{\mathbf{y}} = X\mathbf{w} + b
$$

$$
(X\mathbf{w})_i = \sum_{j=1}^{d} X_{ij} w_j
\qquad
(XW)_{ij} = \sum_{k=1}^{d} X_{ik} W_{kj}
$$

$$
(n \times d)\cdot(d \times m) \rightarrow (n \times m)
\qquad
AB \ne BA \text{ in general}
$$

## 19. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $X$ | the data matrix — rows are examples | `X` |
| $n$ | how many examples | `X.shape[0]` |
| $d$ | how many features | `X.shape[1]` |
| $X_{ij}$ | row $i$, column $j$ | `X[i-1, j-1]` |
| $\mathbb{R}^{n\times d}$ | "all real tables with $n$ rows and $d$ columns" | shape `(n, d)` |
| $X\mathbf{w}$ | one dot product per row | `X @ w` |
| $XW$ | every row against every column | `X @ W` |
| $X^{T}$ | rows and columns swapped | `X.T` |
| $\odot$ | element-wise product | `A * B` |
| $\hat{Y}$ | a matrix of predictions | `Y_hat` |
| rank | how many axes a tensor has | `X.ndim` |
| shape | how long each axis is | `X.shape` |

## 20. One-Minute Explanation

No equations:

> You have a thousand houses and three different ways to value each one. What is a matrix, and why does one multiplication answer all three thousand questions at once?

---

## 21. Exercises

**Level 1 — Observe.** Look at $W$ in §7. One entry is exactly zero. Which valuation does it belong to, what is the model saying by setting it to zero, and what would happen to the insurance column if that entry were $2$ instead? Now look at $\hat{Y}$ in §8: which column is the most expensive, and which house has the largest gap between market value and tax assessment?

**Level 2 — Calculate (by hand, no code).** Using $X$ and $W$ from §7–8, compute $(XW)_{42}$ — house D's insurance value — showing every multiplication. Then compute $X^T X$. What shape is it, and what does its top-left entry $(X^TX)_{11}$ equal? Compare that number with $\frac{1}{n}\sum x_i^2$ from Chapter 2 §10 — what is the relationship, and why is this matrix the one that decided whether Chapter 2 could train at all?

**Level 3 — Derive.** Prove $(AB)^T = B^T A^T$ starting from the entry formula $(AB)_{ij} = \sum_k A_{ik}B_{kj}$. Take it slowly: write what $((AB)^T)_{ij}$ means, then write what $(B^TA^T)_{ij}$ means, and show the two sums are the same sum. Then explain in one sentence why the order *must* reverse — what would go wrong with the shapes if it did not?

**Level 4 — Investigate** (notebook Steps 8–11). Time the loop against `X @ w` for $n = 10,\, 100,\, 1000,\, 10000$ houses. Does the speed ratio stay constant as $n$ grows, or does it change? Then time `X @ W` for a $(1000, 50)$ matrix against fifty separate `X @ w` calls. Is doing all the questions at once faster than doing them one at a time, and can you explain why from §13?

**Level 5 — Design.** The office wants a fourth valuation: *rental yield*, which should depend on area and on the neighbourhood — and neighbourhood is the categorical feature you designed an encoding for in Chapter 2's Level 5. Extend $X$ and $W$ to include it. State the new shapes of $X$, $W$ and $\hat{Y}$. Then answer the harder question: if the encoding needs 50,000 slots for 50,000 neighbourhoods, what is the shape of $W$, how many numbers must be stored, and what does that tell you about why large models are large?

---

## 22. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "A matrix is a 2-D array." | The array is storage. The matrix is the array *with its algebra* — that is what does the work. §3. |
| "`A * B` multiplies matrices." | In NumPy that is element-wise. Matrix multiplication is `A @ B`. They compute different things and often have different shapes. |
| "$AB = BA$, it is just multiplication." | Almost never true, and frequently one of the two is not even defined. §9. |
| "I will fix the shapes once it runs." | Shapes are the cheapest correctness check you have. Use them before running, not after. §11. |
| "The transpose is a formatting detail." | It decides *which axis is summed over*. Getting it wrong silently computes the wrong gradient. §10. |
| "Vectorized code is a different algorithm." | It is the same mathematics, executed differently. If the two disagree, the fast one is wrong. §13. |

## 23. Socratic Questions

1. $X\mathbf{w}$ consumed the feature axis and left the house axis. Which axis does $X^T X$ consume, and what kind of object does that make it — a statement about houses, or about features?
2. We stacked houses as rows and weights as columns. What would have gone wrong if we had stacked *both* as rows?
3. Matrix multiplication is not commutative but it *is* associative: $(AB)C = A(BC)$. Both sides give the same answer — so can it matter which one you compute? Think about the shapes of the intermediate results.
4. A weight of exactly zero means "ignore this feature." What would a *negative* weight mean in the insurance column, and is there a house feature for which that would be sensible?
5. If $X$ is $(n \times d)$ with $n$ much larger than $d$ — many houses, few features — what does that say about whether an exact solution to $X\mathbf{w} = \mathbf{y}$ can exist at all? (You solved a version of this in Chapter 2 §1.)
6. We keep saying the summed axis "disappears." In $XW$, the feature axis vanishes from the output. Is the information in it gone, or just rearranged — and how would you tell the difference?

---

## 24. 🔭 Bridge to Chapter 04

We set out to build a filing cabinet, and we built one. $X$ holds a dataset, $W$ holds a layer's worth of questions, and $XW$ answers all of them at once.

But read §8's result once more, and notice something we walked straight past.

Every house went in as a point with two coordinates — rooms and area. Every house came out as a point with three coordinates — market, insurance, tax. The matrix did not merely *store* numbers. It **took every point in one space and put it somewhere in another.**

That is not filing. That is *movement*. And it raises questions our spreadsheet picture cannot answer: if $W$ moves points around, what does it do to the *shape* of things — does it stretch them, rotate them, flatten them? Does applying $A$ and then $B$ equal applying some single matrix $C$? (§9 said $AB \ne BA$; if matrices are actions, that suddenly makes obvious sense — putting on socks then shoes is not the same as shoes then socks.)

And one question that will decide the architecture of every neural network in this course: **if you chain many matrices together, do you get something genuinely new — or just another matrix?**

> **What does a matrix *do* to space?**

That is where Chapter 04 begins.

➡️ **Next:** [Chapter 04 — A Matrix Can Transform Space](<../Lecture 04 - A Matrix Can Transform Space/blog.md>)
