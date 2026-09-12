# Chapter Format

Every chapter is exactly **two files** in one folder:

```text
Lecture NN - Title/
  blog.md         ← the lecture: theory, mathematics, discovery   (read it)
  notebook.ipynb  ← the laboratory: runnable experiments          (run it)
```

[`LEARNING_CONTRACT.md`](LEARNING_CONTRACT.md) says *what* a lesson must teach.
This document says *how it is written and laid out*.

---

## The Golden Rule

> **Never introduce a formula before explaining the problem that forced humans to invent it.**

The mathematics must feel necessary, not decorative. A reader should finish thinking
*"I discovered this"*, never *"I was handed a formula to memorize."*

Every idea therefore arrives through the same cycle:

```text
real problem → what makes it hard → what a solution would need
  → simplest attempt → where it fails → the next natural question
  → the idea → intuition → tiny numbers → notation → derivation
  → picture → implementation → ML connection → next problem
```

---

## The Discovery Cycle

The cycle above is the unit of writing, **not** the whole chapter. A chapter that teaches
three ideas runs the cycle three times, then closes once.

```text
§ The Problem              ← concrete, visual, relatable; a running example
§ What Would a Solution Need?   ← reason out the requirements before inventing anything
§ First Attempt            ← the simplest idea; show it work, then show it break
§ The Discovery            ← only now does the formula appear
§ Building the Mathematics ← three levels, derivation, shapes
```

Chapter 1, for example, runs it three times: *how do we represent a rule* → *how do we
measure wrongness* → *how do we know which way to move*. Each failure creates the next
question.

### Structure of a chapter

| Part | Sections |
|---|---|
| Opening | `0. The Big Question` (one sentence), `Where We Are` (Previously / Today / Next) |
| Body | 1–4 **discovery cycles**, numbered continuously |
| Seeing it | `The Geometry` — what the idea looks like as a picture |
| Testing it | `The Experiment` — change one variable; `How It Breaks` — remove the assumption |
| Meaning | `Machine Learning Connection`, `Distinctions That Matter` |
| Closing | the mandatory sections below |

### Mandatory closing sections

Every chapter ends with all of these, in order:

1. **What We Discovered** — ideas, not vocabulary.
2. **Mathematics We Built** — every equation introduced, listed.
3. **What Each Symbol Means** — notation table with a `In code` column.
4. **One-Minute Explanation** — explain the chapter with no equations at all.
5. **Exercises** — five levels (below).
6. **Common Mistakes** — wrong reasoning shown, then explained.
7. **Socratic Questions** — *why* questions with no answers given.
8. **Bridge to the Next Chapter** — the unresolved problem this chapter created.

### Exercise levels

| Level | Asks the learner to | Lives in |
|---|---|---|
| 1 — Observe | read a graph or table and interpret it | blog |
| 2 — Calculate | do a small calculation by hand | blog (**Hand Calculation**) |
| 3 — Derive | derive a formula, not just use it | blog (**Mathematical Challenge**) |
| 4 — Investigate | change a variable, predict, run, explain | notebook (Steps 8–11) |
| 5 — Design | invent a solution to an open-ended problem | blog |

Exercises are never repetitive arithmetic. Each one must test a specific misconception or
capability — if you cannot say which, cut it.

---

## Writing rules

### Three levels for every major concept

1. **Child intuition** — an analogy a Class-7 student follows.
2. **Tiny numbers** — 3–5 values, calculated by hand, shown in full.
3. **Abstraction** — the general equation and its notation.

Never jump from story to symbols. Never stay in analogy forever.

### Derive, don't declare

Write the algebra out. `d/dx(x²) = 2x` is not teaching; starting from
$\frac{f(x+h)-f(x)}{h}$ and expanding until $2x + h$ falls out, then asking what happens as
$h \to 0$, is.

### Teach notation as a language

Every symbol gets an English sentence the first time it appears. $\sum_{i=1}^{n} x_i$ reads
*"start at the first value, keep adding, stop at $n$."* Never assume sigma, $\partial$,
$\nabla$ or $\mathbf{bold}$ are already understood.

### Track shapes, always

```text
W ∈ ℝ^{3×2}   x ∈ ℝ^{2×1}   ⟹   Wx ∈ ℝ^{3×1}
```

State why, not just what. Shape reasoning is a habit the course builds from chapter 1.

### Show the tempting wrong idea

Each chapter includes at least one **⚠️ A Tempting Wrong Idea** — an approach a smart
reader would actually try, shown failing for a stated reason.

### Precision over false simplicity

Simple language, correct mathematics. Write *"the gradient gives the direction of steepest
local increase under the usual Euclidean geometry"*, not *"the gradient finds the best
direction."* Never buy simplicity with a false sentence.

### One running example per chapter

Pick it in §1 and return to it in every section. Chapter 1 prices houses; chapter 2 measures
fruit; chapter 16 disambiguates *"it"* in a sentence. Coherence comes from the example, not
from transitions.

### History Lens

A short box per chapter: person, approximate period, the problem they actually faced, why
the idea mattered. Use *"imagine you are Gauss facing…"* framing.
**Never invent quotations** and never attribute modern terminology to historical figures.

### Callouts

> 💡 **Intuition**  🧠 **Think**  📜 **History**  ✏️ **Hand Calculation**
> 🔬 **Experiment**  ⚠️ **Common Mistake**  🎯 **ML Connection**  🔭 **Next Question**

---

## The two-file split

> **Prose, mathematics and derivations live in `blog.md`. Runnable code lives in
> `notebook.ipynb`. Neither file repeats the other.**

| | `blog.md` | `notebook.ipynb` |
|---|---|---|
| Story, motivation, history | ✅ full | ❌ none |
| Derivations, notation | ✅ full | one line restating the equation under test |
| Hand calculation | ✅ step by step | ✅ same numbers, checked by `assert` |
| Runnable programs | ❌ (≤10 illustrative lines) | ✅ all of it |
| Plots | described and predicted | generated |
| Experiments | predicted and explained | executed |

The notebook restates each problem in two or three lines and asks for a prediction. It
never copies the lecture.

### `notebook.ipynb` — the 13-step laboratory

| Step | Cell | Purpose |
|---|---|---|
| — | md | Colab badge (this chapter's own encoded path) |
| 1 | md | **Problem** — the chapter's question in 2–3 lines, link back to `blog.md` |
| 2 | md | **Prediction** — the learner commits to an answer *before* any code runs |
| 3 | md + code | **Intuition** — the smallest possible demonstration |
| 4 | md | **Mathematics** — the equation under test, one line |
| 5 | code | **Manual calculation** — the blog's numbers, verified by `assert` |
| 6 | code | **First implementation** — from scratch, no framework shortcuts |
| 7 | code | **Visualization** — the picture the blog predicted |
| 8 | code | **Experiment** — the controlled run |
| 9 | code | **Change parameters** — exactly one variable moves |
| 10 | md | **Observe** — what actually happened |
| 11 | md | **Explain** — why the mathematics predicted it |
| 12 | code | **Challenge** — levels 4–5, scaffolded |
| 13 | md | **Reflection** — mastery check and the next question |

Code must mirror the mathematics (§ *Code mirrors math* below). A chapter with nothing to
do at a step keeps the step and says why in one line — step numbers mean the same thing in
all 37 chapters.

### Code mirrors math

Never hide the operation being taught behind a library call. Show the explicit form first:

```python
for i in range(rows):          # what the mathematics says
    for j in range(cols):
        ...
y = W @ x                      # what you use once it is understood
```

Then explain why optimized libraries exist.

---

## Coherence rules

1. **Same numbers.** The tiny example in the blog is the exact array in the notebook.
2. **Same names.** A symbol maps to one variable name, and never changes meaning between
   chapters — $\eta$ is always `learning_rate`.
3. **Every numeric claim in the blog is asserted in the notebook.** If the blog says the
   loss falls from 41 to 28.45, the notebook asserts it. This is what stops the two files
   drifting apart silently.
4. **Links both ways.** Blog sections with a lab step link forward; every lab step links
   back to its section.
5. **Predict before run.** Every experiment asks its question before the code cell.
6. **Each section answers a question raised by the previous one.** If it does not, it is in
   the wrong place.

---

## Colab requirements

- Badge points at this chapter's own path on `main`. Folder names contain spaces, so
  encode: space → `%20`, `:` → `%3A`, `'` → `%27`.
- Runs top to bottom on a **fresh CPU runtime** with no edits.
- **No local files** — generate data in code or download by URL.
- `pip install` only in the setup cell, quiet, with a reason.
- Seeds set so printed numbers match the blog exactly.
- Whole notebook runs in **under two minutes**.
- Run it once in Colab and **commit the outputs**, so plots render on GitHub.

## Style

- Title matches the folder name: `# Lecture NN — Title` (em dash).
- H1 once, for the title. Sections are H2.
- Math: `$…$` inline, `$$…$$` display with a blank line either side.
- Diagrams: ```mermaid``` fenced blocks.
- Relative links to paths with spaces must be wrapped: `[text](<blog.md>)`.
- Short paragraphs. Questions. No buzzwords, no motivational filler, no definition dumps.

---

## Coherence audit

Before a chapter is done, check all five:

**Conceptual** — Is there one central problem? Does every section answer the previous
section's question? Is anything introduced before it is motivated?

**Mathematical** — Definitions correct? Every symbol defined? Derivations shown?
Assumptions stated? Shapes right?

**Historical** — Accurate? Properly credited? No fabricated quotes?

**Pedagogical** — Can a motivated Class 7–8 student follow the story? Can an advanced
reader continue into the formal mathematics? Does each abstraction emerge naturally?

**Experimental** — Does the notebook test *this chapter's* ideas? Can the learner predict
outcomes before running? Is every visualization answering a question?

### Definition of done

- [ ] Chapter opens with a problem, not with "in this chapter we will learn".
- [ ] No formula appears before the problem that forced it.
- [ ] Every equation is derived or explicitly motivated.
- [ ] Every symbol is translated into English on first use.
- [ ] One running example carries the whole chapter.
- [ ] At least one tempting wrong idea is shown failing.
- [ ] All mandatory closing sections are present.
- [ ] Notebook runs clean on a fresh runtime, outputs committed.
- [ ] Numbers, symbols and variable names match across both files.
- [ ] Every numeric claim in the blog is asserted in code.
