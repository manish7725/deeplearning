# Blog 16 — Attention: What Should I Look At?

Imagine this sentence:

**The animal did not cross the road because it was tired.**

To understand what “it” means, we need to pay attention to the relevant words.

Attention gives a neural network a mathematical way to decide which pieces of information are important to each other.

## 1. Query, Key, Value

Attention uses three ideas:

- **Query** — What am I looking for?
- **Key** — What information does each item advertise?
- **Value** — What information should I receive if I pay attention to it?

These are transformed versions of the original representations.

## 2. A simple score

Suppose a query is `q` and a key is `k`.

A basic similarity score is their dot product:

`score = q · k`

If the vectors point in similar directions, the score can be large.

So the model can use the score to decide where to focus.

## 3. Softmax turns scores into weights

Suppose the scores are:

`[2, 1, 0]`

Softmax converts them into positive numbers that add up to 1.

Conceptually, the largest score receives the largest attention weight.

So instead of saying:

“Look at word number 2.”

we can say:

“Give word 2 a lot of attention, word 1 some attention, and word 3 a little.”

## 4. The mathematical formula

A common scaled dot-product attention equation is:

`Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V`

It looks intimidating, but each part has a job:

- `QKᵀ` compares queries with keys.
- `√dₖ` keeps scores at a useful scale.
- `softmax` turns scores into weights.
- multiplying by `V` combines information according to those weights.

## 5. Why attention changed deep learning

Instead of forcing information through one tiny sequential memory, attention allows many relationships to be considered directly.

That makes it especially powerful for language and other structured data.

> **Attention is a learned mathematical spotlight: it decides which information should influence the current representation.**