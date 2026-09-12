# Blog 16 — Attention: What Should I Look At?

Consider this sentence:

> **“The animal didn't cross the road because it was tired.”**

What does “it” refer to?

A good language model needs to examine the surrounding words and decide which information matters.

That is the intuition behind **attention**.

---

## 1. The central question

For a token, ask:

> **Which other tokens should influence my representation right now?**

Attention turns that question into mathematics.

---

## 2. Queries, keys and values

For each token representation, a neural network produces three vectors:

- **Query ($Q$):** what am I looking for?
- **Key ($K$):** what information do I offer for matching?
- **Value ($V$):** what information should be passed along?

The projections are typically

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

where $X$ contains the input token representations.

---

## 3. Compare a query with keys

The simplest comparison is a dot product:

$$
QK^T
$$

Each entry measures how strongly a query matches a key.

For example, if

$$
Q=\begin{bmatrix}1&2\end{bmatrix}
$$

and

$$
K=\begin{bmatrix}2&1\end{bmatrix}
$$

then

$$
QK^T=1(2)+2(1)=4
$$

A larger score means a stronger raw match.

---

## 4. Why divide by $\sqrt{d_k}$?

Scaled dot-product attention uses

$$
\frac{QK^T}{\sqrt{d_k}}
$$

where $d_k$ is the key dimension.

The scaling helps keep dot-product magnitudes in a useful range as dimensionality grows, which helps the softmax behave better during optimization.

---

## 5. Softmax turns scores into weights

Suppose the scores are

$$
[2,1,0]
$$

Softmax is

$$
\operatorname{softmax}(z_i)=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

The outputs are positive and sum to 1.

So attention scores become weights that describe how strongly to combine the values.

---

## 6. The complete formula

The standard scaled dot-product attention equation is

$$
\boxed{
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

This looks intimidating, but read it left to right:

1. compare queries with keys;
2. scale the scores;
3. turn scores into weights with softmax;
4. use the weights to mix the values.

---

## 7. A visual mental model

```mermaid
flowchart LR
    X[Token representations] --> Q[Queries]
    X --> K[Keys]
    X --> V[Values]
    Q --> S[QK^T / sqrt(dk)]
    K --> S
    S --> P[Softmax weights]
    P --> O[Weighted values]
    V --> O
```

Attention is therefore a learned information-routing mechanism.

---

## 8. Self-attention

When the queries, keys and values all come from the same sequence, we call it **self-attention**.

Every token can compare itself with other tokens and build a new representation using information from the sequence.

This is a major reason Transformers can model long-range relationships effectively.

---

## 9. Causal attention

For language generation, a token should not normally look into the future.

For the sequence

```text
I love deep learning
```

when predicting the next token, the model can use previous tokens but should not peek at future target tokens.

A causal mask enforces this restriction.

```text
        I  love  deep  learning
I       ✓   ×     ×      ×
love    ✓   ✓     ×      ×
deep    ✓   ✓     ✓      ×
```

The exact mask convention depends on implementation, but the principle is future information is blocked.

---

## 10. PyTorch

Modern PyTorch provides attention-related building blocks. A conceptual example is:

```python
import torch
import torch.nn.functional as F

Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)

scores = Q @ K.transpose(-2, -1)
scores = scores / (K.size(-1) ** 0.5)
weights = F.softmax(scores, dim=-1)
out = weights @ V

print(out.shape)
```

This code exposes the mathematics directly.

---

## Think Like a Scientist 🧠

Suppose the query is the word “bank”.

Imagine the sentence:

> “I deposited money at the bank.”

Which words should receive strong attention?

Now change it to:

> “We sat beside the river bank.”

Which words should matter now?

The surrounding context changes the useful information.

---

## What you should remember

> **Attention lets each representation selectively combine information from other representations.**

The central equation is

$$
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This single idea became the foundation of the Transformer architecture.

> **Next: Transformers — building an entire architecture around attention.**
