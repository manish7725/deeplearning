# Blog 37 — How Can We Teach a Model What Humans Prefer?

> **Deep Learning from First Principles — written for a Class 7 mind, but with the mathematics kept honest.**

Suppose an AI can write two answers.

```text
Answer A → correct but confusing
Answer B → correct and easy to understand
```

A human might prefer B.

But ordinary next-token training does not automatically know our preference.

So we need another idea:

> **Teach the model which behaviors are preferred.**

This connects deep learning with **reinforcement learning and policy optimization**.

---

# 1. Prediction is not preference

Language-model training often teaches:

> “Predict the next token.”

But a useful assistant needs more:

```text
helpful
safe
clear
relevant
```

These are preferences about behavior.

---

# 2. Compare answers

Suppose a human sees:

```text
Question
  ↓
Model
  ├── Answer A
  └── Answer B
```

The human chooses B.

We now have a preference signal:

$$
B>A
$$

Collect many such comparisons and we can train another model to predict which answer humans are likely to prefer.

That model is often called a **reward model**.

---

# 3. Reward

Imagine an AI playing a game.

Good move:

$$
+1
$$

Bad move:

$$
-1
$$

The AI learns a policy—a strategy for choosing actions.

For language models, the “action” can be generating tokens.

The reward can represent how desirable the final answer is.

---

# 4. Policy optimization

A policy is a rule for choosing actions.

In simple notation:

$$
\pi(a|s)
$$

means:

> probability of choosing action $a$ in situation $s$.

The goal is to adjust the policy so that good outcomes become more likely.

A family of methods called **policy-gradient methods** does this using gradients.

One famous algorithm is **PPO**, or Proximal Policy Optimization.

The important beginner idea is not the full PPO equation.

It is:

```text
choose action
   ↓
observe reward
   ↓
calculate learning signal
   ↓
update policy
   ↓
try again
```

---

# 5. Why “proximal”?

Imagine a student changes their answer to every question dramatically after one piece of feedback.

That could be dangerous.

PPO tries to make policy updates more controlled.

Very roughly:

> **Improve the policy, but do not let one update change behavior too wildly.**

---

# 6. From language model to assistant

A simplified pipeline can look like:

```text
pretrained language model
          ↓
collect human preferences
          ↓
learn reward signal
          ↓
optimize model behavior
          ↓
more preferred answers
```

Modern alignment methods can be much more complicated, but this gives us the basic picture.

---

# 🧠 A school analogy

Imagine a basketball player.

First they learn the rules of basketball.

Then a coach says:

> “That move is legal, but this move is better.”

The player already knows how to play.

The coach is shaping behavior toward a preference.

That is similar to the distinction between **learning to predict** and **learning preferred behavior**.

---

# 🧪 Think Like a Scientist

Create five pairs of chatbot answers.

For each pair, ask a human to choose:

```text
A or B?
```

Train a tiny classifier to predict the preference.

Now you have built a toy reward model.

You do not need a giant language model to understand the idea.

---

# 🧠 What you should remember

1. Prediction and preference are different problems.
2. Human comparisons can provide preference signals.
3. A reward model can learn to predict those preferences.
4. A policy describes how a model chooses actions.
5. Policy optimization changes behavior toward higher reward.
6. PPO is designed to make policy updates more controlled.

> **First teach a model to understand language. Then teach it how we prefer it to behave.**

---

# 🧪 Hands-on challenge

Create ten pairs of answers to the same questions.

Ask several people to choose the better answer.

Build a small preference dataset:

| Question | Answer A | Answer B | Preferred |
|---|---|---|---|
| ... | ... | ... | A/B |

Train a simple classifier.

Your experiment is a tiny version of a much larger idea: **learning from human preferences**.