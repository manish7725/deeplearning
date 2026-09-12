# Blog 33 — What Happens After a Model Is Trained?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

We have spent many lessons training models.

But imagine the model is finally trained.

Now someone asks it a question.

What happens?

The answer is **inference**.

Inference is the process of using a trained model to produce an answer.

But for modern models, inference can be much more interesting than simply pressing “run.”

---

# 1. The simplest inference

Suppose our model is:

$$
\hat y=f_\theta(x)
$$

During training, $\theta$ changes.

During ordinary inference, $\theta$ is fixed.

We give the model a new input:

```text
input
 ↓
trained model
 ↓
prediction
```

---

# 2. Classification inference

Suppose a model outputs:

```text
cat      0.70
dog      0.20
rabbit   0.10
```

A simple classifier chooses the largest value:

```text
cat
```

But language models create a different problem.

They need to choose what comes next.

---

# 3. Language-model inference

Suppose the sentence is:

```text
The sky is
```

The model may produce probabilities such as:

```text
blue   0.60
clear  0.20
grey   0.10
...
```

We need to turn probabilities into a token.

That is a **decoding** problem.

---

# 4. Greedy decoding

The simplest strategy is:

> Pick the most likely next token.

```text
step 1 → blue
step 2 → ...
```

It is simple and fast.

But always choosing the largest probability can sometimes produce boring or locally sensible text that becomes globally poor.

---

# 5. Beam search

Instead of keeping one possible answer, keep several.

Imagine exploring a maze.

```text
start
 ├── path A
 ├── path B
 └── path C
```

At each step, keep the best few paths.

This is the basic idea of **beam search**.

It can be useful for structured generation tasks.

---

# 6. Sampling

Instead of always choosing the largest probability, we can sample according to the model's distribution.

That introduces controlled variation.

For example:

```text
blue   0.60
clear  0.20
grey   0.10
red    0.01
```

A sampling method may sometimes choose `clear` even though `blue` is most likely.

That can make generation less repetitive.

---

# 7. Reasoning at inference time

Modern systems can sometimes use extra computation while answering.

Instead of:

```text
question → immediate answer
```

they can use:

```text
question
  ↓
consider possibilities
  ↓
search / reason
  ↓
answer
```

MIT's inference lecture discusses methods including beam search, chain-of-thought, in-context learning, test-time training and search-based techniques. citeturn1search2turn1search12

For a young learner, the key idea is:

> **Inference can include a strategy for deciding how to use the model, not just one forward pass.**

---

# 8. In-context learning

A language model can sometimes change its behavior based on examples included in the prompt.

```text
Example 1 → answer
Example 2 → answer
Example 3 → answer

New question → ?
```

The model is not necessarily changing its stored weights.

The context changes what it does during inference.

---

# 🧠 Training vs inference

| Training | Inference |
|---|---|
| learns parameters | uses parameters |
| calculates gradients | normally no gradient update |
| changes weights | weights usually fixed |
| expensive learning process | answer-generation process |

Both are important.

---

# 🧪 Think Like a Scientist

Take a tiny language model or classifier.

Compare:

```text
greedy decoding
sampling
beam search
```

Generate multiple outputs from the same prompt.

Ask:

> How can the same trained model produce different answers simply because we changed the inference strategy?

---

# 🧠 What you should remember

1. Inference means using a trained model.
2. Classification can choose the most likely output.
3. Language models must choose tokens repeatedly.
4. Greedy decoding picks the current best option.
5. Beam search keeps multiple candidate paths.
6. Sampling introduces controlled randomness.
7. Modern inference can use additional computation or search.
8. In-context learning can change model behavior without changing its weights.

> **Training teaches the model. Inference decides how to use what the model has learned.**

---

# 🧪 Hands-on challenge

Use a small language model and generate text using different decoding settings.

Save:

```text
prompt
method
output
```

Compare the outputs.

Then explain which method you would choose for:

- a creative story
- a fixed-format answer
- a translation

There is no single best inference strategy for every task.