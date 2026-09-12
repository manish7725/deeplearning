# Blog 18 — How Does a Language Model Learn to Predict Text?

A language model can be introduced with a surprisingly simple question:

> “Given what I have seen so far, what is likely to come next?”

## 1. Predict the next token

Suppose the text is:

`The sun rises in the ...`

Possible next words might include:

`east`, `morning`, `sky`.

During training, the model is given the actual next token and learns to assign it a high probability.

## 2. Probabilities

Instead of producing only one answer, a model can produce a probability distribution.

For a tiny vocabulary:

```text
sun   0.05
east  0.70
moon  0.10
fish  0.15
```

The probabilities add up to 1.

The model predicts a distribution over possible next tokens.

## 3. Softmax

The final neural-network scores are called logits.

Softmax converts logits into probabilities:

`pᵢ = eᶻⁱ / Σⱼ eᶻʲ`

You do not need to memorize the formula yet.

The important idea is that softmax turns a list of arbitrary scores into positive numbers that sum to one.

## 4. The model makes mistakes

Suppose the correct next token is `east`, but the model gives it probability 0.1.

That should produce a larger loss than if the model had assigned probability 0.9.

A common training objective is **cross-entropy loss**.

For one correct class:

`L = -log(p_correct)`

If `p_correct` is small, the loss is large.
If `p_correct` is close to 1, the loss is small.

## 5. Training

The training loop is familiar now:

`text → tokens → vectors → Transformer → probabilities → loss → gradients → parameter updates`

Millions or billions of parameter updates can gradually improve the model.

## 6. Prediction after training

Once trained, the model can repeatedly predict the next token.

It can generate:

`token 1 → token 2 → token 3 → ...`

This simple repeated process can produce surprisingly rich text.

## 7. The important lesson

A language model does not need a human to write a rule for every sentence.

It learns statistical structure from examples.

> **A language model learns a numerical model of patterns in sequences and uses that model to predict what comes next.**