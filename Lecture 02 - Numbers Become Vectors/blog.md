# Lecture 02 — Numbers Become Vectors

> **The Big Question:** How can a list of numbers become a single mathematical object we can compute with?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2002%20-%20Numbers%20Become%20Vectors/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** A machine discovered $\hat{y} = wx + b$ from four house sales — ₹2 lakh per room, ₹1 lakh for the plot — by measuring its own wrongness and stepping downhill.
**Today:** That model describes a house with **one** number. We find out what breaks when it needs three, and build the object that fixes it.
**Next:** One house will be one vector. But we have thousands of houses, and we want to price them all at once — which forces us into **matrices**.

---

## 1. The Problem: Two Houses, Same Rooms, Different Prices

Your model from Chapter 1 works. The office is pleased. Then four new sales come in, and this time somebody recorded the floor area as well:

| House | Rooms | Area (sq ft) | Price (₹ lakh) |
|---|---:|---:|---:|
| A | 2 | 800 | 9 |
| B | 2 | 1200 | 11 |
| C | 3 | 900 | 11.5 |
| D | 4 | 1600 | 17 |

Look at A and B.

**Same number of rooms. Two lakh rupees apart.**

Now recall exactly what our model is. $\hat{y} = wx + b$ takes one input and returns one output. Feed it $x = 2$ and it returns *one* number — whatever $w \cdot 2 + b$ happens to be. It must give A and B the same price.

So it is wrong about at least one of them, and no amount of gradient descent can fix that. There is no value of $w$ and no value of $b$ that makes one input produce two different outputs. **The failure is not in the training. It is in the representation.**

The machine cannot see the area, because we never gave it a way to hold two measurements at once.

---

## 2. What Would a Solution Need?

We need a way to describe a house that:

1. **Holds several measurements together**, as one thing we can pass around.
2. **Keeps them distinguishable.** Rooms and square feet must not blur into each other.
3. **Lets each measurement carry its own influence.** A room is worth ₹2 lakh; a square foot is worth far less. The model must be able to learn a *different* weight for each.
4. **Scales to any number of measurements** — three today, three hundred later, without rewriting the mathematics.

Requirement 3 is the one that kills the obvious shortcut.

---

## 3. First Attempt: Squash Them Into One Number

We already have a working one-input model. So let us just combine rooms and area into a single "house score" and feed that in.

Add them: $2 + 800 = 802$ for house A, $2 + 1200 = 1202$ for house B.

It works, in the narrow sense that A and B now differ. But look closely at what that number is made of. Area contributes hundreds; rooms contribute two or three. The room count is **0.2%** of the score. We have technically kept both measurements and effectively thrown one away.

Worse, this locks the exchange rate. By adding them we declared that *one extra room is worth exactly one extra square foot* — that gaining a room and gaining a square foot are equally valuable events. Nobody believes that. And there is no dial in this scheme to correct it, because the squashing happened *before* the model ever saw the numbers.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Combine the features into a single score, then reuse the model we already have."*
>
> Any collapsing of many numbers into one — adding, averaging, "total score" — decides the relative importance of the measurements **for** the machine, before learning starts. That is precisely the decision we wanted the machine to make. Requirement 3, violated.

The fix must keep the measurements apart, and give each one its own dial:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + b
$$

With $w_1 = 2$ (₹ lakh per room), $w_2 = 0.005$ (₹ lakh per square foot) and $b = 1$:

$$
\text{House A: } 2(2) + 0.005(800) + 1 = 4 + 4 + 1 = 9 \;\checkmark
$$
$$
\text{House B: } 2(2) + 0.005(1200) + 1 = 4 + 6 + 1 = 11 \;\checkmark
$$

Both correct, from the same rule. The machine can now explain A and B, because area has its own dial.

This works — and it still does not scale. Three measurements means three terms. A photograph has a million pixels:

$$
\hat{y} = w_1x_1 + w_2x_2 + w_3x_3 + \cdots + w_{1000000}x_{1000000} + b
$$

Nobody can write that, and no mathematics can be done on it in that form. **We do not need a different idea — we need a different notation.**

---

## 4. The Discovery: A Vector

Stop writing the measurements as separate named quantities. Put them in a box, in a fixed order, and give the box one name:

$$
\mathbf{x}_A = \begin{bmatrix} 2 \\ 800 \end{bmatrix}
\qquad
\mathbf{x}_B = \begin{bmatrix} 2 \\ 1200 \end{bmatrix}
$$

This is a **vector**. The order is part of the meaning: slot 1 is always rooms, slot 2 is always area. Change the order and you change what the object says.

| Level | The same idea |
|---|---|
| 💡 **Intuition** | A backpack with labelled pockets. One pocket holds rooms, one holds area. You carry the whole backpack around as one item, and you can always reach into a specific pocket. |
| ✏️ **Numbers** | $[2, 800]$ means *2 rooms, 800 square feet*. Not 802 of anything. Two facts, kept separate, travelling together. |
| 🎓 **Abstraction** | $\mathbf{x} \in \mathbb{R}^{d}$ — an ordered list of $d$ real numbers. $d$ is the **dimension**. |

The bold $\mathbf{x}$ is not decoration. It is a promise that this symbol holds several numbers, while plain $x$ holds one. $x_2$ means *the second component of $\mathbf{x}$* — here, 800.

The weights get the same treatment. There is one dial per measurement, so the dials form a vector of the same length:

$$
\mathbf{w} = \begin{bmatrix} 2 \\ 0.005 \end{bmatrix}
$$

> 📜 **History Lens — Grassmann, Hamilton, Gibbs**
>
> Descartes gave us coordinates in 1637: a point on a plane as a pair of numbers. For two centuries that pair stayed a *location* — not something you could add or multiply as a single object.
>
> The change came from two directions in the 1840s. William Rowan Hamilton, trying to extend complex numbers to three dimensions, invented quaternions in 1843. Hermann Grassmann, a schoolteacher in Stettin, published *Die lineale Ausdehnungslehre* in 1844 — a general theory of "extended magnitudes" that contains most of modern linear algebra. It was so far ahead of its notation that almost nobody read it; Grassmann largely gave up mathematics for Sanskrit scholarship, where he became genuinely famous.
>
> The notation we actually use came later, in the 1880s, when Josiah Willard Gibbs at Yale and Oliver Heaviside in England independently stripped Hamilton's quaternions down to the practical vector algebra — dot products, cross products, bold letters — that physics and, a century later, machine learning would run on.
>
> The lesson worth keeping: **the idea was available for forty years and went nowhere until someone found the right way to write it down.** Notation is not bookkeeping. It is what makes thought possible at scale.

---

## 5. Vectors Are Also Places

Here is the second face of the same object. Write a two-measurement house as a pair, and it is a **point on a map**:

```text
  area
  1600 │              • D
       │
  1200 │     • B
       │
   900 │         • C
   800 │  • A
       │
       └──────────────────── rooms
          2    3    4
```

House A sits at $(2, 800)$. House B sits directly above it — same rooms, more area. The geometry *is* the data: houses that are alike sit near each other, and the direction you move in tells you what changed.

An equally valid reading is an **arrow** from the origin to that point. Point or arrow is a matter of what you are about to do: points are for *positions* (this house), arrows are for *changes* (this house has 400 sq ft more than that one). Same numbers, two readings.

With three measurements you get a point in 3-D space. With 784 — the pixels of a small digit image — you get a point in $\mathbb{R}^{784}$, which nobody can picture. That is fine. **The algebra does not care how many dimensions there are, and the intuition from two dimensions keeps working.** This is the single most useful habit in the whole course: reason in 2-D, compute in 784-D.

---

## 6. Two Operations We Actually Need

Before the main event, two operations that fall out of the picture immediately.

**Addition** — add matching slots:

$$
\begin{bmatrix} 2 \\ 800 \end{bmatrix} + \begin{bmatrix} 1 \\ 400 \end{bmatrix} = \begin{bmatrix} 3 \\ 1200 \end{bmatrix}
$$

*"Add one room and 400 square feet."* Geometrically: walk along the first arrow, then continue along the second.

**Scalar multiplication** — stretch every slot by the same factor:

$$
2 \begin{bmatrix} 2 \\ 800 \end{bmatrix} = \begin{bmatrix} 4 \\ 1600 \end{bmatrix}
$$

*"Twice the house."* Geometrically: the arrow keeps its direction and changes length. Multiply by $-1$ and it points the opposite way.

Note what we did **not** define: multiplying two vectors slot-by-slot. $[2, 800] \times [1, 400] = [2, 320000]$ is a computation you *can* do — NumPy will do it happily — but nothing in our problem asks for it. Ask instead what the pricing problem actually needs, and a different operation appears.

---

## 7. The Discovery: The Dot Product

Return to the rule that worked:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + b
$$

Read the pattern in the first two terms out loud: *multiply each weight by its matching measurement, then add up all the results.* Every term pairs slot $i$ of $\mathbf{w}$ with slot $i$ of $\mathbf{x}$.

That operation — pair up, multiply, sum — is exactly what pricing a house requires. So we name it. The **dot product** of two vectors of the same length:

$$
\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{d} w_i x_i
$$

Read the sigma as: *"go through the slots one at a time, multiply the two entries you find there, keep a running total."* For house A:

$$
\mathbf{w} \cdot \mathbf{x}_A = (2)(2) + (0.005)(800) = 4 + 4 = 8
$$

And the entire model collapses to:

$$
\boxed{\;\hat{y} = \mathbf{w} \cdot \mathbf{x} + b\;}
$$

$$
\hat{y}_A = 8 + 1 = 9 \;\checkmark
$$

Look at what that box survives. Two measurements or two million, the formula does not change — only the length of the vectors does. Requirement 4, satisfied. Chapter 1's $\hat{y} = wx + b$ turns out to be this same formula in the special case $d = 1$.

> 💡 **Intuition** — A dot product is a **weighted vote**. Each measurement votes for a price; the weight says how loud that measurement's voice is; the dot product tallies the votes into one number.

Watch the shapes, because this is the property that makes it the right tool:

```text
   w: (d,)        x: (d,)
        ↓  pair, multiply, sum
        w · x : a single number
```

Two vectors go in; **one** number comes out. We wanted one price. That is not a coincidence — it is why this operation, and not slot-wise multiplication, is the one the problem demanded.

---

## 8. How Big Is a Vector?

New question, and it will matter enormously by Chapter 15: **how different are two houses?**

To answer that we first need the length of a single arrow. In two dimensions this is Pythagoras. The arrow to $(3, 4)$ is the hypotenuse of a right triangle with sides 3 and 4:

$$
\|\mathbf{v}\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

In three dimensions, apply Pythagoras twice — once in the base plane, once for the height — and the pattern is identical: sum the squares, take the root. It keeps going, unchanged, forever:

$$
\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{d} x_i^2}
$$

This is the **norm** — the length of the arrow, the distance from the origin to the point. Read $\|\mathbf{x}\|$ as *"the size of $\mathbf{x}$."*

Notice it is built from a dot product with itself:

$$
\mathbf{x} \cdot \mathbf{x} = \sum x_i^2 = \|\mathbf{x}\|^2
$$

And the **distance between two houses** is the length of the arrow from one to the other:

$$
d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|
$$

---

## 9. How Similar Are Two Houses?

Take three houses, measuring area in hundreds of square feet to keep the arithmetic visible:

$$
\mathbf{p} = \begin{bmatrix} 1 \\ 4 \end{bmatrix}
\qquad
\mathbf{q} = \begin{bmatrix} 2 \\ 8 \end{bmatrix}
\qquad
\mathbf{r} = \begin{bmatrix} 4 \\ 1 \end{bmatrix}
$$

$\mathbf{p}$ is a small flat: 1 room, 400 sq ft. $\mathbf{q}$ is exactly twice $\mathbf{p}$ — 2 rooms, 800 sq ft, the same *kind* of property at double the size. $\mathbf{r}$ is strange: 4 rooms crammed into 100 sq ft.

Ask distance which pair is most alike:

$$
\|\mathbf{p} - \mathbf{q}\| = \left\| \begin{bmatrix} -1 \\ -4 \end{bmatrix} \right\| = \sqrt{17} \approx 4.12
\qquad
\|\mathbf{p} - \mathbf{r}\| = \left\| \begin{bmatrix} -3 \\ 3 \end{bmatrix} \right\| = \sqrt{18} \approx 4.24
$$

Distance says $\mathbf{p}$ is about equally close to both. But $\mathbf{q}$ is the *same proportions* as $\mathbf{p}$ and $\mathbf{r}$ is a completely different sort of building. Distance measured the wrong thing: it is dominated by **size**, and we asked about **character**.

What actually distinguishes them on the map is the **direction** each arrow points. $\mathbf{p}$ and $\mathbf{q}$ lie along the same ray from the origin; $\mathbf{r}$ points somewhere else entirely. So measure the angle between them.

### Deriving the connection between angle and dot product

Here is the satisfying part. Take the triangle formed by $\mathbf{u}$, $\mathbf{v}$ and the arrow $\mathbf{u} - \mathbf{v}$ joining their tips, with angle $\theta$ between $\mathbf{u}$ and $\mathbf{v}$. The law of cosines — Pythagoras generalized to non-right triangles — says:

$$
\|\mathbf{u}-\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
$$

Now expand that same left-hand side using only §8's fact that $\|\mathbf{a}\|^2 = \mathbf{a}\cdot\mathbf{a}$, multiplying out as ordinary algebra:

$$
\|\mathbf{u}-\mathbf{v}\|^2 = (\mathbf{u}-\mathbf{v})\cdot(\mathbf{u}-\mathbf{v})
= \mathbf{u}\cdot\mathbf{u} - 2\,\mathbf{u}\cdot\mathbf{v} + \mathbf{v}\cdot\mathbf{v}
= \|\mathbf{u}\|^2 - 2\,\mathbf{u}\cdot\mathbf{v} + \|\mathbf{v}\|^2
$$

Two expressions for one quantity. Set them equal; $\|\mathbf{u}\|^2$ and $\|\mathbf{v}\|^2$ cancel from both sides:

$$
-2\,\mathbf{u}\cdot\mathbf{v} = -2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
$$

$$
\boxed{\;\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta\;}
$$

We invented the dot product in §7 for a bookkeeping reason — tallying weighted votes. It turns out to have been measuring **angles** the entire time. Nobody designed that; it fell out of the algebra.

Rearranged, it gives a similarity score that ignores size:

$$
\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|}
$$

Apply it to our houses:

$$
\cos(\mathbf{p}, \mathbf{q}) = \frac{(1)(2)+(4)(8)}{\sqrt{17}\cdot\sqrt{68}} = \frac{34}{\sqrt{1156}} = \frac{34}{34} = \mathbf{1}
$$

$$
\cos(\mathbf{p}, \mathbf{r}) = \frac{(1)(4)+(4)(1)}{\sqrt{17}\cdot\sqrt{17}} = \frac{8}{17} \approx \mathbf{0.47}
$$

Exactly 1 for the pair that shares proportions, 0.47 for the odd one. This is **cosine similarity**, and it reads on a fixed scale no matter how large the vectors are:

| $\cos\theta$ | Angle | Meaning |
|---:|---:|---|
| $1$ | $0°$ | same direction — same character |
| $0$ | $90°$ | **orthogonal** — unrelated |
| $-1$ | $180°$ | opposite |

> 🔭 **Next Question, deferred** — Hold onto this. In Chapter 15 we will represent the *meaning of a word* as a vector, and "how similar are these two words" will be answered with exactly this formula. The reason cosine is the right tool there is the reason it was right here: we care about direction, not magnitude.

---

## 10. 🔬 The Experiment: When Features Live on Different Scales

Chapter 1 ended with a challenge: rescale the input from $[1,2,3,4]$ to $[10,20,30,40]$, find the largest safe learning rate again, and explain the shift. Here is that question, now unavoidable — because our two features differ in scale by a factor of 400 all by themselves.

> 🧠 **Predict before you read on.** We train $\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$ on the four houses, with rooms in the range 2–4 and area in the range 800–1600. Does gradient descent find $\mathbf{w} = [2,\, 0.005]$? If it struggles, is it too slow, or does it explode?

Chapter 1 §10 showed that the steepness of the loss valley along a feature is governed by $\frac{1}{n}\sum x_i^2$. Compute it for each of ours:

$$
\text{rooms: } \frac{2^2+2^2+3^2+4^2}{4} = 8.25
\qquad
\text{area: } \frac{800^2+1200^2+900^2+1600^2}{4} = 1{,}362{,}500
$$

One direction of the valley is roughly **165,000 times** steeper than the other. The landscape is not a bowl — it is a long, almost flat canyon with near-vertical walls.

Now the consequence, and it is brutal. The learning rate is a *single* number applied to every direction at once. It must be small enough to survive the steepest wall — anything larger explodes, exactly as in Chapter 1 §14. But that same tiny step, applied along the nearly-flat floor, moves the rooms weight almost not at all:

| Features | Steepest direction | Flattest direction | Ratio | Largest safe $\eta$ | Result after 2000 steps |
|---|---:|---:|---:|---:|---|
| raw | 2,725,016 | 0.756 | 3,604,714 | $\approx 7\times10^{-7}$ | loss still **1.76**; $\mathbf{w} \approx [0.002,\, 0.011]$ — barely moved |
| scaled | 2.07 | 0.024 | 85 | $\approx 0.5$ | loss **0.0000003**; correct answer |

There is **no** learning rate that works on the raw features. Too large and it explodes off the canyon wall; too small and the rooms weight never moves. That ratio has a name — the **condition number** of the problem — and 3.6 million is a catastrophe.

The fix is almost embarrassingly simple. Divide each feature by its largest value so that everything lands in $[0, 1]$:

$$
\text{rooms} \to \frac{\text{rooms}}{4}, \qquad \text{area} \to \frac{\text{area}}{1600}
$$

The condition number falls from 3,604,714 to 85. Training with $\eta = 0.1$ now converges, and recovers

$$
\mathbf{w}_{\text{scaled}} \approx [8.00,\; 8.00], \qquad b \approx 1.00
$$

Undo the scaling to read the answer in real units — $8.00 / 4 = 2.00$ ₹ lakh per room, $8.00/1600 = 0.005$ ₹ lakh per square foot, $b = 1$ ₹ lakh for the plot. **Exactly the true rule.**

And as in Chapter 1 §14, the mathematics says in advance where it will break. Counting the bias as a third direction, the steepest curvature of the scaled problem is $4.0028$, which puts the stability limit at $2/4.0028 = 0.4997$. Run it: $\eta = 0.45$ converges, $\eta = 0.5$ blows up. The threshold was predictable to two decimal places before a single step was taken.

> 💡 This is why every serious model normalizes its inputs, and it is not a trick or a convention. Feature scaling changes the *shape of the loss landscape* — and Chapter 1 taught us that the shape of the landscape is the whole problem.

---

## 11. How It Breaks

| Failure | What it looks like | Why |
|---|---|---|
| **Mismatched scales** | no usable learning rate | §10. The condition number explodes. |
| **Numbers that are not quantities** | confident nonsense | Encode cities as Delhi=1, Mumbai=2, Chennai=3 and the geometry claims Mumbai is *between* the other two, and that Delhi + Chennai = 2 × Mumbai. The arithmetic is meaningless but runs silently. |
| **Duplicated information** | unstable, arbitrary weights | Give the model area in sq ft *and* in sq m. Infinitely many $(w_1, w_2)$ pairs now give identical predictions; the flattest direction becomes exactly flat. |
| **Missing measurement** | irreducible error | If price depends on the neighbourhood and you never recorded it, no vector of the ones you did record can recover it. §1's failure, in general form. |
| **Wrong similarity** | "similar" items are nothing alike | §9. Distance and cosine answer different questions; using one where you meant the other fails quietly. |

> ⚠️ **Common Mistake** — believing that adding features always helps. Every feature adds a dimension, and dimensions cost data: points spread out and *everything* looks far from everything else in high dimensions. Chapter 25 gives this its proper name.

---

## 12. 🎯 Machine Learning Connection

Vectors are not a chapter of the course. They are the format of the entire subject.

| Thing | As a vector |
|---|---|
| A house | $[\text{rooms},\, \text{area},\, \text{age}]$ |
| A greyscale digit image | $[p_1, \ldots, p_{784}]$ — one slot per pixel |
| A word | a learned $[e_1, \ldots, e_{768}]$ — Chapter 15 |
| A sentence | a sequence of such vectors — Chapter 17 |
| One layer's weights | one vector per neuron — Chapter 5 |
| Every parameter in a model | one enormous $\theta$ — Chapter 1's update rule, unchanged |

And the operation runs everywhere too. A single neuron computes $\mathbf{w}\cdot\mathbf{x} + b$; attention scores in a transformer are dot products between queries and keys; the "similar images" in a search engine are nearest neighbours by cosine. **Learn the dot product properly and you have learned the arithmetic that most of deep learning spends its time doing.**

---

## 13. Distinctions That Matter

| | |
|---|---|
| **List** — any ordered collection | **Vector** — a list you can add, scale and dot, whose order carries meaning |
| **Dot product** $\mathbf{w}\cdot\mathbf{x}$ — two vectors in, **one number** out | **Element-wise product** $\mathbf{w} \odot \mathbf{x}$ — two vectors in, a **vector** out |
| **Dimension** $d$ — how many measurements | **Shape** `(d,)` — how the numbers are arranged in memory |
| **Distance** $\|\mathbf{u}-\mathbf{v}\|$ — how far apart, sensitive to size | **Cosine** — how aligned, blind to size |
| **Point** — a position (this house) | **Arrow** — a change (this much bigger than that) |
| $x$ — one number | $\mathbf{x}$ — several, in fixed order |

---

## 14. What We Discovered

1. Chapter 1's model failed not because of bad training but because its *representation* could not hold two measurements.
2. Collapsing several measurements into one number silently decides their relative importance — the very thing the machine was supposed to learn.
3. A vector keeps measurements separate and travelling together, and it is simultaneously a point in space.
4. The dot product was not imported from geometry; it is what "weighted vote" looks like when written compactly — and it makes the model's form independent of how many features there are.
5. That same operation turns out to measure angles, which we proved from the law of cosines.
6. Length comes from Pythagoras and does not change shape as dimensions grow.
7. Distance measures size difference; cosine measures character. They are different questions.
8. Feature scales control the shape of the loss landscape, and mismatched scales can make a problem untrainable at *any* learning rate.

## 15. Mathematics We Built

$$
\mathbf{x} \in \mathbb{R}^{d}
\qquad
\hat{y} = \mathbf{w}\cdot\mathbf{x} + b
\qquad
\mathbf{w}\cdot\mathbf{x} = \sum_{i=1}^{d} w_i x_i
$$

$$
\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{d} x_i^2} = \sqrt{\mathbf{x}\cdot\mathbf{x}}
\qquad
d(\mathbf{u},\mathbf{v}) = \|\mathbf{u}-\mathbf{v}\|
$$

$$
\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta
\qquad
\cos\theta = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

## 16. What Each Symbol Means

| Symbol | English | In code |
|---|---|---|
| $\mathbf{x}$ | a vector — several numbers in fixed order | `x` |
| $x_i$ | the $i$-th slot of $\mathbf{x}$ | `x[i-1]` |
| $d$ | dimension — how many slots | `len(x)` |
| $\mathbb{R}^d$ | "the set of all $d$-number lists" | shape `(d,)` |
| $\mathbf{w}\cdot\mathbf{x}$ | pair up, multiply, sum → one number | `w @ x` |
| $\|\mathbf{x}\|$ | length of the arrow | `np.linalg.norm(x)` |
| $\odot$ | element-wise product → a vector | `w * x` |
| $\theta$ | angle between two vectors | — |
| $\cos\theta$ | alignment, from $-1$ to $1$ | `cosine(u, v)` |

## 17. One-Minute Explanation

With no equations:

> Why can't you describe a house with one number, and what does an arrow have to do with it?

---

## 18. Exercises

**Level 1 — Observe.** Look at the map in §5. Houses A and B sit one directly above the other. What does "directly above" mean about the two houses, and what would "directly to the right" have meant? Which pair on that map is closest by distance, and is that the pair you would call most similar?

**Level 2 — Calculate (by hand).** For $\mathbf{w} = [2,\, 0.005]$ and $b = 1$, compute $\hat{y}$ for houses C $[3, 900]$ and D $[4, 1600]$. Then compute $\|\mathbf{x}_C\|$, $\|\mathbf{x}_D\|$ and the distance between them. Finally: change house C's area from 900 to 901 and recompute the price. By how much did it move, and which number in $\mathbf{w}$ did you just measure?

**Level 3 — Derive.** Prove that $\mathbf{u}\cdot\mathbf{v} = \mathbf{v}\cdot\mathbf{u}$ straight from the sigma definition. Then prove $\|c\,\mathbf{u}\| = |c|\,\|\mathbf{u}\|$ for a scalar $c$, and explain why the absolute value is necessary. Finally, use the boxed result of §9 to show that cosine similarity can never exceed 1 — and say what it means geometrically when it equals exactly 1.

**Level 4 — Investigate** (notebook Steps 8–11). Find the largest learning rate that works on the raw features, then on the scaled features, and confirm the ratio matches the ratio of $\frac{1}{n}\sum x_i^2$ between them. Then try scaling *only* the area and leaving rooms raw. Does partial scaling help, hurt, or do nothing — and does the condition number predict what you see?

**Level 5 — Design.** The office now records the neighbourhood: `Andheri`, `Bandra`, `Colaba`. You must turn that into numbers a dot product can consume. Numbering them 1, 2, 3 fails for the reason in §11. Design an encoding that does not lie about the geometry. How many slots does it need? What is the distance between any two neighbourhoods under your scheme, and is that the right answer? What breaks if the city has 50,000 neighbourhoods?

---

## 19. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| "A vector is just a Python list." | A list has no notion of length, angle or addition. The operations are the point; the storage is not. |
| "`w * x` computes the dot product." | In NumPy that is element-wise and returns a *vector*. The dot product is `w @ x` and returns one number. Shapes catch this instantly. |
| "More features means a better model." | Each feature adds a dimension and a way to mislead — duplicated, unscaled or meaningless ones all hurt. §11. |
| "Distance is similarity." | Distance is dominated by magnitude. §9's $\mathbf{p}$ and $\mathbf{q}$ are the same kind of house and far apart. |
| "Scaling features is just a preprocessing convention." | It changes the geometry of the loss landscape and can decide whether the model trains at all. §10. |
| "A 784-dimensional vector is unimaginable, so I cannot reason about it." | The algebra is identical to 2-D. Picture two dimensions, compute in 784. |

## 20. Socratic Questions

1. We chose to put rooms in slot 1 and area in slot 2. Does the model care which order we chose? Does *anything* change if we swap them consistently everywhere?
2. Cosine similarity ignores magnitude. When would that be exactly the wrong property to want?
3. Two vectors are orthogonal when their dot product is 0. What does it mean for two *features* to be orthogonal across a dataset — and why might you want that?
4. If normalizing features helps this much, why not normalize the *target* prices too? What would change, and what would you have to undo afterwards?
5. In §10 the ratio between the steepest and flattest directions was the thing that hurt. Can a landscape be badly conditioned even when every feature is on the same scale?
6. We treat a photograph as a point in $\mathbb{R}^{784}$, which throws away the fact that pixel 1 sits next to pixel 2. What does the model lose — and which chapter do you think gives it back?

---

## 21. 🔭 Bridge to Chapter 03

We can now describe one house as one vector, and price it with one dot product.

But the office has four houses on the desk, and a real dataset has millions. Pricing them means running the same dot product against the same $\mathbf{w}$, over and over — a loop that gets no shorter no matter how neatly we write a single step.

And a second pressure is coming. In Chapter 5 a network will want **many different weight vectors** at once — one per neuron, each asking a different question about the same house. That is a whole collection of $\mathbf{w}$'s to organize.

Stack vectors in rows and you have a table of numbers. Give that table its own algebra and one symbol can hold an entire dataset, or an entire layer.

> **How do we hold many vectors as a single object, and compute with all of them in one operation?**

That object is the **matrix**, and it is where Chapter 03 begins.

➡️ **Next:** [Chapter 03 — Matrices: The Spreadsheet of Mathematics](<../Lecture 03 - Matrices: The Spreadsheet of Mathematics/blog.md>)
