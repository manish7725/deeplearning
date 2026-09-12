# Lecture 17 — Decision Trees

> **The Big Question:** Can a classifier learn a useful rule by repeatedly asking simple yes-or-no questions?

▶️ **Run the code:** [Open in Colab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2017%20-%20Decision%20Trees/notebook.ipynb) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are

**Previously:** Nearest neighbours classified a new house by looking at nearby houses, but high-dimensional distance can become unreliable.
**Today:** We replace distance with a sequence of simple questions such as “Is the area less than 12?” and let the data choose which question to ask first.
**Next:** One tree can memorize peculiar details. We will ask how several imperfect trees can become a stronger model.

---

## 1. The Problem: A House Needs a Decision, Not a Coordinate

Return to our familiar property table:

| House | Rooms | Area (hundreds sq ft) | Class |
|---|---:|---:|---|
| A | 2 | 8 | Apartment |
| B | 2 | 10 | Apartment |
| C | 3 | 12 | Apartment |
| D | 4 | 16 | Villa |
| E | 5 | 18 | Villa |
| F | 3 | 20 | Farmhouse |

A new house arrives with 3 rooms and 11 hundred square feet.

Nearest neighbours ask *“Which old houses are close?”*

A decision tree asks a different question:

> **What simple question would split these classes apart?**

Perhaps:

```text
Is area < 14?
├── yes → mostly apartments
└── no  → mostly villas/farmhouses
```

Then we can ask another question inside each branch.

### What Would a Solution Need?

A useful tree must decide:

1. Which feature should we ask about?
2. Where should we split that feature?
3. Which split makes the children more “pure”?
4. When should we stop splitting?

The word **pure** needs an exact mathematical meaning.

---

## 2. First Attempt: Choose the Feature We Like

A person might look at the table and simply choose *area* because it visually separates the classes.

That works on this tiny dataset. But humans can be biased, and with dozens of features we need a rule that the machine can calculate.

> ⚠️ **A Tempting Wrong Idea**
>
> *"Choose the feature that looks most useful."*
>
> Visual intuition is useful for exploration, but it is not a reproducible learning algorithm. Two people may choose different questions.

So we need to measure how mixed a group of labels is **before and after** a split.

---

## 3. The Discovery: Measure Impurity

Imagine a box containing ten property labels.

If all ten are **Villa**, the box is perfectly pure.

If the box contains five Villa and five Apartment, the box is mixed.

One common measure of impurity is **Gini impurity**:

$$
\boxed{G=1-\sum_{k=1}^{K}p_k^2}
$$

where $p_k$ is the fraction of examples belonging to class $k$.

Read the formula in words:

> Square each class's share, add those squares, then subtract the result from one.

For a pure box with

$$
[1,0,0],
$$

we get

$$
G=1-(1^2+0^2+0^2)=0.
$$

No uncertainty remains.

For a box with two equally common classes,

$$
[0.5,0.5],
$$

we get

$$
G=1-(0.25+0.25)=0.5.
$$

The more mixed the labels, the larger the impurity.

---

## 4. Three Levels

| Level | The same idea |
|---|---|
| 💡 **Intuition** | A fruit bowl is pure if every fruit is the same kind. A mixed bowl has higher impurity. |
| ✏️ **Tiny numbers** | Two classes split 50/50 give $G=0.5$; a pure class gives $G=0$. |
| 🎓 **Abstraction** | $G=1-\sum_kp_k^2$ measures label mixture in a node. |

The tree will now search for questions that turn one mixed box into purer boxes.

---

## 5. The Discovery: Compare Parent and Children

Suppose the current node has 10 houses.

A candidate split makes two children:

- Left child: 4 houses, impurity $0.0$
- Right child: 6 houses, impurity $0.333$

We cannot simply add the two impurities because the right child contains more examples.

We use a weighted average:

$$
G_{children}
=
\frac{n_L}{n}G_L+\frac{n_R}{n}G_R.
$$

Then the improvement from the split is

$$
\boxed{\text{gain}=G_{parent}-G_{children}}.
$$

A good split makes the children substantially purer, so the gain is large.

This is the machine's answer to the question:

> **Which question should I ask first?**

Choose the split with the largest impurity reduction.

---

## 6. A Hand Calculation

Suppose our parent node has:

```text
Apartment = 4
Villa     = 4
Farmhouse = 2
```

So

$$
\mathbf p=[0.4,0.4,0.2].
$$

The parent impurity is

$$
G_P=1-(0.4^2+0.4^2+0.2^2)
=1-(0.16+0.16+0.04)
=0.64.
$$

Now split it into:

**Left:** 4 apartments.

$$
G_L=0.
$$

**Right:** 4 villas and 2 farmhouses.

$$
\mathbf p_R=[0,4/6,2/6].
$$

Therefore

$$
G_R=1-\left(\frac46\right)^2-\left(\frac26\right)^2
=1-\frac{16}{36}-\frac{4}{36}
=\frac{16}{36}\approx0.444.
$$

The weighted child impurity is

$$
G_C=\frac{4}{10}(0)+\frac{6}{10}(0.444)
\approx0.2667.
$$

So the gain is

$$
0.64-0.2667\approx0.3733.
$$

That is a substantial improvement.

---

## 7. Entropy: Another Way to Measure Mixture

Gini impurity is not the only possible measure.

Chapter 15 already gave us the logarithm. We can use it again to measure uncertainty:

$$
\boxed{H=-\sum_k p_k\log_2 p_k}
$$

This is **entropy**.

For a perfectly pure node,

$$
H=0.
$$

For two equally likely classes,

$$
H=-(0.5\log_2 0.5+0.5\log_2 0.5)=1.
$$

A decision tree can choose splits using **information gain**:

$$
\boxed{IG=H_{parent}-H_{children}}.
$$

Gini and entropy often produce similar tree structures. The important idea is not memorizing which one is “the true” impurity. It is understanding the general pattern:

> **Choose the question that reduces uncertainty the most.**

---

## 8. Building the Tree Recursively

Once we know how to score a split, the construction becomes almost mechanical:

```text
1. Start with all training examples.
2. Find the best split.
3. Send examples left or right.
4. Repeat inside each child.
5. Stop when the node is pure or a stopping rule says enough.
```

This repetition is called **recursion**.

A tree is therefore not magic. It is one simple idea applied again and again to smaller subsets of the data.

---

## 9. Continuous Features Need Thresholds

What if area is a real number such as 12.7?

A tree can ask questions like:

$$
\text{Is area}<12.5?
$$

or

$$
\text{Is area}<15.0?
$$

For a sorted feature, candidate thresholds can be considered between observed values.

For example, if the observed areas are

$$
8,10,12,16,18,20,
$$

possible thresholds include 9, 11, 14, 17, and 19.

The tree tests candidates and keeps the one with the best impurity reduction.

This is why trees can model nonlinear decision regions without us manually creating curved features.

---

## 10. Geometry: Trees Cut Space Into Boxes

A linear classifier from Chapters 13–15 makes a boundary such as

$$
w_1x_1+w_2x_2+b=0.
$$

A decision tree makes axis-aligned cuts:

```text
          area
           ↑
       Villa | Villa
             |
   Apartment | Farmhouse
   ----------+----------→ rooms
```

Every split slices the space along one feature axis.

After several questions, the feature space is partitioned into rectangular regions.

This is a crucial geometric difference:

> **Linear models create flat boundaries. Trees create collections of axis-aligned regions.**

---

## 11. Why Trees Can Model Nonlinear Rules

Consider the XOR pattern:

```text
class B   •       •   class B

class A   •       •   class A
```

with labels alternating across a square. One straight line cannot separate the classes.

A tree can first ask about one coordinate, then ask about the other coordinate inside each branch.

The tree does not need a curved equation. It builds the shape by combining many simple rectangles.

---

## 12. The Dangerous Power: A Tree Can Memorize

Suppose we keep splitting until every leaf contains exactly one training example.

Training accuracy can become 100%.

That sounds perfect — but it means the tree has learned every tiny accident in the training set.

This is **overfitting**.

We can control it by limiting things such as:

- maximum depth,
- minimum number of samples in a leaf,
- minimum number of samples required to split,
- maximum number of leaves.

These are not mathematical necessities. They are choices that control how much complexity we allow the tree to use.

---

## 13. 📜 History Lens — Quinlan and ID3

Imagine you are Ross Quinlan in the 1980s, interested in turning examples into understandable rules.

Decision-tree systems such as ID3 made an important practical idea concrete: **a model can learn by asking the most informative question available at each step.**

This connected machine learning to something humans already understand well — a game of twenty questions.

A good tree is therefore a question-and-answer conversation with the data:

```text
Question → split → question → split → prediction
```

Modern tree algorithms have become much more sophisticated, but that core idea survives.

---

## 14. 🔬 The Experiment

Open the [lab](https://colab.research.google.com/github/manish7725/deeplearning/blob/main/Lecture%2017%20-%20Decision%20Trees/notebook.ipynb).

Predict before running:

1. What is the Gini impurity of a 50/50 two-class node?
2. Which split should the toy dataset prefer: one that makes a pure child, or one that leaves both children equally mixed?
3. What happens to training accuracy as maximum depth increases?
4. What happens to test accuracy after the tree becomes too deep?

---

## 15. How It Breaks

| Failure | What it looks like | Why it happens |
|---|---|---|
| Very deep tree | Training accuracy near 100%, poor test accuracy | The tree memorizes noise |
| Tiny leaves | Unstable predictions | Few examples determine a region |
| Greedy split | Good local choice, not globally optimal tree | Most algorithms choose the best current split |
| Too shallow | Both training and test accuracy poor | Important structure remains unmodeled |
| Small dataset | Unstable thresholds | A few examples strongly affect impurity estimates |

---

## 16. Shapes

For $N$ examples and $d$ features:

```text
X : (N, d)
y : (N,)
feature column : (N,)
left mask : (N,)
right mask: (N,)
```

A binary split can be represented by a Boolean mask such as

```python
left = X[:, feature] < threshold
right = ~left
```

The key shape is unchanged: a split chooses a subset of the original rows. The feature matrix does not need to become a new mathematical object for every node; each recursive call simply receives a smaller set of rows.

---

## 17. 🎯 Machine Learning Connection

Decision trees are especially useful when the relationship between features and labels is naturally expressed as rules.

They have several practical strengths:

- little preprocessing is often required,
- nonlinear relationships are easy to represent,
- interactions between features appear naturally through depth,
- decisions can be inspected as a sequence of questions.

But a single tree is often unstable: a small change in data can produce a different set of splits. That instability is exactly what the next chapter will exploit.

---

## 18. Distinctions That Matter

| Confusable pair | Difference |
|---|---|
| Node vs leaf | A node can still split; a leaf is a final prediction region. |
| Gini impurity vs information gain | Gini measures mixture; information gain measures reduction in entropy after a split. |
| Depth vs number of nodes | Depth is the longest root-to-leaf path; a tree can have many nodes without being equally deep everywhere. |
| Training accuracy vs generalization | A tree can memorize training examples without learning a useful rule for unseen data. |
| Greedy vs globally optimal | Choosing the best current split does not guarantee the globally smallest tree. |

---

## 19. What We Discovered

1. A classifier can be built as a sequence of simple questions.
2. A node is “good” when its labels are pure or nearly pure.
3. Gini impurity and entropy quantify label uncertainty.
4. A split is valuable when it reduces impurity.
5. Repeating the best split recursively creates a decision tree.
6. Axis-aligned cuts let trees represent nonlinear decision regions.
7. Unlimited growth turns a tree into a memorization machine.

---

## 20. Mathematics We Built

$$
G=1-\sum_kp_k^2
$$

$$
G_{children}=\sum_c\frac{n_c}{n}G_c
$$

$$
\text{gain}=G_{parent}-G_{children}
$$

$$
H=-\sum_kp_k\log_2p_k
$$

$$
IG=H_{parent}-H_{children}
$$

---

## 21. What Each Symbol Means

| Symbol | Meaning | In code |
|---|---|---|
| $p_k$ | fraction of class $k$ in a node | `class_prob` |
| $G$ | Gini impurity | `gini` |
| $H$ | entropy | `entropy` |
| $n$ | examples in parent node | `n_node` |
| $n_c$ | examples in child $c$ | `n_child` |
| $IG$ | information gain | `information_gain` |
| threshold | candidate cutoff | `threshold` |

---

## 22. One-Minute Explanation

A decision tree learns a sequence of questions. At every node it tries possible feature thresholds and chooses one that makes the resulting groups purer. Gini impurity or entropy tells us how mixed a node is; impurity reduction tells us which question is useful. Repeating the process builds a tree that can carve complicated regions out of feature space. If we let the tree grow without restraint, it can simply memorize the training data.

---

## 23. Exercises

### Level 1 — Observe
Look at two candidate splits and identify which produces purer child nodes.

### Level 2 — Calculate
Compute Gini impurity for class proportions $[0.7,0.3]$.

### Level 3 — Derive
Derive the weighted-child impurity formula from the idea that each example should contribute according to the fraction of examples in its child.

### Level 4 — Investigate
Train trees with maximum depths 1, 2, 4, 8, and unrestricted. Plot training and validation accuracy.

### Level 5 — Design
Design a stopping rule for a decision tree that has only 30 training examples. Explain what kind of overfitting your rule is trying to prevent.

---

## 24. Common Mistakes

| Mistake | Why it is wrong |
|---|---|
| Thinking a pure node must contain one class in the whole dataset | Purity is local to the current node. |
| Comparing child impurity without weighting by size | A child with 2 examples should not count the same as one with 98. |
| Assuming deeper always means better | Extra depth can fit noise. |
| Thinking trees need scaled features for basic threshold splits | Monotonic rescaling changes the numeric threshold but not the ordering. |
| Assuming the greedy split is globally optimal | Local gain does not solve the global tree optimization problem. |

---

## 25. Socratic Questions

Why does squaring class probabilities appear in Gini impurity?

Why should child impurity be weighted by the number of examples?

Why can a tree model XOR even though each individual split is simple?

Why can two nearly identical training sets produce different tree structures?

Why would combining many unstable trees possibly make the final prediction more stable?

---

## 🔭 Bridge to Chapter 18

We now have a powerful but unstable learner. A single decision tree can be clever, interpretable, and nonlinear — but it can also chase noise.

What if we deliberately build **many different trees**, let them make independent mistakes, and combine their votes?

That leads to forests and boosting: the idea that many imperfect learners can form a stronger one.