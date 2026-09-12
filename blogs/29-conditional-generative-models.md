# Blog 29 — Can We Tell a Generative Model What to Create?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Imagine asking an artist:

> “Draw a red car.”

That is different from saying:

> “Draw anything.”

The first request contains a **condition**.

Modern generative models can also be conditioned.

This gives us:

```text
condition + noise/information
          ↓
      generator
          ↓
       output
```

---

# 1. Generation without a condition

A generative model learns something about a dataset.

For example:

```text
many dog pictures
       ↓
model learns patterns
       ↓
new dog-like picture
```

But we may want control.

---

# 2. Add a condition

Suppose $c$ is a condition.

The model can generate:

$$
x=f(z,c)
$$

where:

- $z$ is a source of variation
- $c$ tells the model what we want
- $x$ is the generated result

For example:

$$
c=\text{“cat”}
$$

might encourage the model to generate a cat.

---

# 3. Different kinds of conditions

The condition could be:

```text
text
class label
image
another image
style
attributes
```

So we can build systems such as:

```text
text → image
image → image
text → text
image → text
```

MIT's generative-model lecture explicitly covers conditional GANs, conditional VAEs and conditional diffusion models, including paired and unpaired translation and text-to-image generation. citeturn1search8turn1search3

---

# 4. A classroom analogy

Imagine a teacher says:

> “Draw an animal.”

Many answers are possible.

Then the teacher says:

> “Draw a bird.”

The possibilities become smaller.

Then:

> “Draw a blue bird flying over a tree.”

The condition gives the generator more information about the desired result.

---

# 5. Conditional diffusion — the big idea

Diffusion models start from noise and gradually create structure.

A simplified picture is:

```text
random noise
     ↓
remove some noise
     ↓
remove more noise
     ↓
more structure
     ↓
image
```

With a condition:

```text
random noise + “red car”
          ↓
      denoising steps
          ↓
       red car
```

The exact mathematics is deeper, but the idea is simple:

> **The condition guides the generation process.**

---

# 🧠 Why this matters

Without conditioning:

```text
“Make something.”
```

With conditioning:

```text
“Make something matching this request.”
```

That difference is central to useful generative AI.

---

# 🧪 Think Like a Scientist

Train or use a small conditional generator.

Try:

```text
condition = digit 0
condition = digit 1
condition = digit 2
```

Generate many examples for each condition.

Ask:

> Does changing only the condition change the generated result in the way we expect?

---

# 🧠 What you should remember

1. A generative model can create new examples.
2. A condition gives the generation process extra information.
3. Conditional models can be controlled using text, labels, images or other signals.
4. Conditional diffusion guides denoising with the condition.
5. Generation becomes much more useful when we can control what is produced.

> **Generation answers “what can I create?” Conditioning adds “what do I want?”**

---

# 🧪 Hands-on challenge

Use a digit dataset.

Train a conditional model that receives:

```text
noise + digit label
```

Generate ten examples for each digit.

Put them in a grid.

Can you tell which column corresponds to which condition?

If yes, your condition is influencing the generator.