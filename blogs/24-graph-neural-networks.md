# Blog 24 — When Data Looks Like a Network: Graph Neural Networks

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Look around you.

You are not just a list of features.

You are connected to:

```text
friends
teachers
family
school
```

The connections matter.

Now imagine a social network, a road map, a molecule, or a computer network.

All of them can be represented as a **graph**.

And graphs need a neural network designed for connections.

Enter the **Graph Neural Network (GNN)**.

---

# 1. What is a graph?

A graph contains:

- **nodes** — things
- **edges** — relationships between things

For example:

```text
Alice ─── Bob
  │       │
  │       │
Charlie ─ David
```

The people are nodes.

The friendship connections are edges.

A graph can be written mathematically using an adjacency matrix.

For four nodes:

$$
A=\begin{bmatrix}
0&1&1&0\\
1&0&0&1\\
1&0&0&1\\
0&1&1&0
\end{bmatrix}
$$

A `1` means two nodes are connected.

---

# 2. Why a normal neural network struggles

Suppose node 1 has a feature:

$$
x_1=10
$$

But what if the important information is in its neighbors?

```text
       neighbor
          ↓
neighbor → A → neighbor
          ↑
       neighbor
```

A GNN lets a node **listen to its neighbors**.

This is called **message passing**.

---

# 3. Message passing

Imagine every student tells their neighbors something about themselves.

Then every student combines what they heard.

Mathematically, a simple GNN layer can look like:

$$
h_i' = \sigma\left(W_1h_i + W_2\sum_{j\in N(i)}h_j\right)
$$

Do not fear the symbols.

Read them as a story:

```text
new information for i
=
my information
+
information from my neighbors
→
activation
```

That is the heart of message passing.

---

# 4. A classroom example

Imagine four students.

Each student knows their own score.

But the teacher wants to understand the classroom group.

A student can learn something from nearby classmates:

```text
my score
   +
neighbors' scores
   ↓
new representation
```

After one layer, a node knows about its immediate neighbors.

After two layers, information can travel two steps away.

```text
Layer 1 → 1-hop information
Layer 2 → 2-hop information
Layer 3 → 3-hop information
```

This gives GNNs a natural way to spread information through a network.

MIT's course treats graph neural networks as a separate architecture family, alongside CNNs, RNNs and Transformers. citeturn0search3turn1search1

---

# 5. Where are GNNs useful?

### 🧪 Molecules

```text
atoms → nodes
bonds → edges
```

A model can learn patterns related to molecular properties.

### 🚗 Roads

```text
junctions → nodes
roads → edges
```

### 👥 Social networks

```text
people → nodes
friendships → edges
```

### 🖥 Computer networks

```text
servers → nodes
connections → edges
```

The data structure changes, but the idea remains:

> **Things + relationships.**

---

# 6. The important trick: aggregation

Suppose node A has three neighbors.

We might collect their information:

$$
h_1,h_2,h_3
$$

and add them:

$$
m_A=h_1+h_2+h_3
$$

Or average them:

$$
m_A=\frac{h_1+h_2+h_3}{3}
$$

Then combine that message with A's own information.

The aggregation should not depend on the arbitrary order in which we list the neighbors.

That is why operations such as sum, mean or max are useful.

---

# 🧠 The big picture

We now have a useful architecture map:

```text
Grid/image      → CNN
Sequence        → RNN / Transformer
Graph           → GNN
```

This is not a rule saying one architecture can only solve one problem.

It is a reminder that **data structure influences architecture design**.

---

# 🧪 Think Like a Scientist

Take a school bus route.

Represent:

- bus stops as nodes
- roads as edges
- number of passengers as node features
- travel time as edge features

Now ask:

> Could a model predict whether a bus stop will become crowded by looking at nearby stops?

You have just designed a graph-learning problem.

---

# 🧠 What you should remember

1. A graph contains nodes and edges.
2. Nodes represent things.
3. Edges represent relationships.
4. GNNs let nodes gather information from neighbors.
5. Message passing is the core idea.
6. Aggregation lets us combine neighbor information without depending on arbitrary ordering.
7. Graphs appear in molecules, roads, social networks and computer systems.

> **When relationships are part of the data, the relationships themselves become something a neural network can learn from.**

---

# 🧪 Hands-on challenge

Create a graph with five nodes.

Give every node one number.

Implement one message-passing step using:

$$
h_i'=\text{ReLU}\left(h_i+\sum_{j\in N(i)}h_j\right)
$$

Print every node before and after the update.

Then add a second layer.

Watch how information travels farther through the graph.