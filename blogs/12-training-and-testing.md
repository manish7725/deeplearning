# Blog 12 — How Do We Know If Our Model Really Learned?

Imagine a student memorizes the answers to 100 questions.

You give exactly those 100 questions in an exam, and the student gets 100/100.

Did the student understand the subject?

Maybe. But we need new questions to know.

Machine learning has the same problem.

## 1. Training data

The examples used to adjust the model's parameters are called the **training data**.

The model sees these examples many times while learning.

## 2. Test data

We keep some examples hidden during training.

Later, we show them to the trained model.

These examples form part of the **test data**.

The test asks a powerful question:

> Can the model work on examples it has never seen before?

## 3. Memorization versus understanding

Suppose a model sees:

`2 + 2 = 4`

`3 + 3 = 6`

`4 + 4 = 8`

If it has learned the underlying pattern, it should also understand:

`10 + 10 = 20`

But if it merely memorized the training examples, it may fail.

## 4. Overfitting

When a model performs extremely well on training examples but poorly on new examples, we call this **overfitting**.

It is like memorizing the textbook without understanding the ideas.

A good model should learn useful patterns rather than simply remember individual examples.

## 5. Validation data

In practical machine learning, we often divide data into:

- training set — learn parameters
- validation set — choose models and settings
- test set — final evaluation

The exact split depends on the problem.

## 6. Why this matters

A model is not valuable because it can reproduce yesterday's answers.

It is valuable because it can make useful predictions tomorrow.

That is called **generalization**.

> **The real goal of machine learning is not memorization. It is useful generalization to new examples.**