---
name: write-chapter
description: Write or rewrite a chapter of this deep-learning course as a discovery narrative - blog.md plus notebook.ipynb - to the standard set by chapters 01-03. Use whenever asked to write, rewrite, draft or continue a chapter/lecture, or when asked what a chapter should contain. Carries the authoring rules, the 60-chapter spine, and the verification workflow.
---

# Writing a Chapter

This repository is one continuous argument from Class-7 intuition to research-level
understanding. Each chapter is **two files in one folder**:

```text
Lecture NN - Title/
  blog.md         ← the lecture: theory, mathematics, discovery   (read it)
  notebook.ipynb  ← the laboratory: runnable experiments          (run it)
```

Chapters **01, 02 and 03** are the reference implementations. Read the one before and after
your target chapter before writing a word.

---

## 1. The Golden Rule

> **Never introduce a formula before explaining the problem that forced humans to invent it.**

The reader must finish thinking *"I discovered this"*, never *"I was handed a formula."* If a
formula appears before the problem it solves, the chapter is wrong and must be restructured,
not patched.

## 2. The Discovery Cycle

The unit of writing is not the chapter — it is the cycle. A chapter teaching three ideas runs
the cycle three times, then closes once.

```text
The Problem              ← concrete, visual, uses the running example
What Would a Solution Need?   ← reason out requirements BEFORE inventing anything
First Attempt            ← the simplest idea; show it work, then show it break
The Discovery            ← only now does the formula appear
Building the Mathematics ← three levels, derivation, shapes
```

Chapter 01 runs it three times (how to represent a rule → how to measure wrongness → how to
know which way to move). Chapter 02 runs it twice. Each failure creates the next question.

### Chapter skeleton

```markdown
# Lecture NN — Title

> **The Big Question:** one sentence.

▶️ **Run the code:** [Open in Colab](<encoded-url>) · [`notebook.ipynb`](<notebook.ipynb>)

## Where We Are
**Previously:** … **Today:** … **Next:** …

## 1..K   — the discovery cycles, numbered continuously
## The Geometry        — what the idea looks like as a picture
## 🔬 The Experiment   — change one variable; predict first
## How It Breaks       — failure table: what it looks like / why
## Shapes              — track every dimension (from Ch 03 onward)
## 🎯 Machine Learning Connection
## Distinctions That Matter    — two-column table of confusable pairs
## What We Discovered          — numbered, ideas not vocabulary
## Mathematics We Built        — every equation introduced
## What Each Symbol Means      — table with an "In code" column
## One-Minute Explanation      — explain it with no equations
## Exercises                   — five levels, below
## Common Mistakes             — table: mistake / why it is wrong
## Socratic Questions          — why-questions, answers NOT given
## 🔭 Bridge to Chapter NN+1
```

### Exercise levels

| Level | Asks the learner to | Lives in |
|---|---|---|
| 1 — Observe | read a graph or table and interpret it | blog |
| 2 — Calculate | do a small calculation by hand | blog |
| 3 — Derive | derive a formula, not use it | blog |
| 4 — Investigate | change a variable, predict, run, explain | notebook |
| 5 — Design | invent a solution to an open problem | blog |

Every exercise must test a specific misconception or capability. If you cannot name which, cut it.

## 3. Writing rules

**Three levels for every major concept** — child intuition (an analogy), tiny numbers
(3–5 values, calculated in full), then abstraction. Never jump story → symbols. Never stay in
analogy forever. Chapters 02 and 03 present this as a literal three-row table.

**Derive, don't declare.** Write the algebra out. Chapter 01 derives the slope $15(w-2)$ by
expanding $\frac{L(w+h)-L(w)}{h}$ until only $7.5h$ is left to discard — earning the update
rule without assuming calculus. Chapter 02 derives $\mathbf{u}\cdot\mathbf{v} =
\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ from the law of cosines.

**Teach notation as a language.** Every symbol gets an English sentence on first use.
$\sum_{i=1}^{n}x_i$ reads *"start at the first value, keep adding, stop at $n$."* Never assume
$\sum$, $\partial$, $\nabla$ or $\mathbf{bold}$ are known.

**Track shapes, always.** State why, not just what. From Chapter 03 onward, show that shapes
alone can rule out a wrong formula before any arithmetic.

**Show the tempting wrong idea.** At least one ⚠️ **A Tempting Wrong Idea** per chapter — an
approach a smart reader would actually try, shown failing for a stated reason. Chapter 02's is
"combine the features into one score"; Chapter 01's is "averaging signed errors."

**Precision over false simplicity.** Simple language, correct mathematics. Write *"the gradient
gives the direction of steepest local increase under the usual Euclidean geometry."* Never buy
simplicity with a false sentence. Say when something is a **choice** (squared error) rather
than a law.

**One running example per chapter**, chosen in §1 and returned to in every section. Continue
the previous chapter's example where possible — Chapters 01–03 all price the same houses.

**History Lens** — one box per chapter: person, approximate period, the problem they actually
faced, why it mattered. Use *"imagine you are Gauss facing…"* framing. **Never invent
quotations.** If a famous quote's provenance is disputed, say so (Chapter 01 does this with the
Samuel quotation) or quote only what is verifiable.

**Callouts:** 💡 Intuition · 🧠 Think · 📜 History · ✏️ Hand Calculation · 🔬 Experiment ·
⚠️ Common Mistake · 🎯 ML Connection · 🔭 Next Question

**Style:** H1 once for the title (`# Lecture NN — Title`, em dash, matching the folder). Sections
are H2. `$…$` inline, `$$…$$` display with blank lines either side. ```mermaid``` for diagrams.
Short paragraphs, questions, no buzzwords, no motivational filler, no definition dumps.

## 4. The two-file split

> **Prose, mathematics and derivations live in `blog.md`. Runnable code lives in
> `notebook.ipynb`. Neither file repeats the other.**

| | `blog.md` | `notebook.ipynb` |
|---|---|---|
| Story, motivation, history | ✅ full | ❌ none |
| Derivations, notation | ✅ full | one line restating the equation under test |
| Hand calculation | ✅ step by step | ✅ same numbers, checked by `assert` |
| Runnable programs | ❌ (≤10 illustrative lines) | ✅ all of it |
| Plots | described and predicted | generated |

### The 13-step laboratory

| Step | Cell | Purpose |
|---|---|---|
| — | md | Colab badge (this chapter's own encoded path) |
| 1 | md | **Problem** — the question in 2–3 lines, link back to `blog.md` |
| 2 | md | **Prediction** — learner commits to answers *before* any code runs |
| 3 | code | **Intuition** — the smallest demonstration |
| 4 | md | **Mathematics** — the equations under test, one line each |
| 5 | code | **Manual calculation** — the blog's numbers, verified by `assert` (split 5a/5b/5c freely) |
| 6 | code | **First implementation** — from scratch, no framework shortcuts |
| 7 | code | **Visualization** — the picture the blog predicted |
| 8 | code | **Experiment** — the controlled run |
| 9 | code | **Change parameters** — exactly one variable moves |
| 10 | md | **Observe** — what actually happened, against Step 2 |
| 11 | md | **Explain** — why the mathematics predicted it |
| 12 | code | **Challenge** — levels 4–5, scaffolded with `# YOUR CODE HERE` |
| 13 | md | **Reflection** — mastery checklist and the next question |

Markdown cells in the notebook are at most five lines. Step numbers mean the same thing in
every chapter; a chapter with nothing to do at a step keeps it and says why in one line.

**Code mirrors math.** Never hide the operation being taught behind a library call. Show the
explicit loop version first, assert it equals the vectorized version, then explain why
optimized libraries exist.

## 5. Coherence rules

1. **Same numbers.** The tiny example in the blog is the exact array in the notebook.
2. **Same names.** One symbol maps to one variable name, never changing meaning between
   chapters — $\eta$ is always `learning_rate`.
3. **Every numeric claim in the blog is asserted in the notebook.** If the blog says the loss
   falls from 41 to 28.45, the notebook asserts it. This is what stops the two files drifting.
4. **Links both ways.** Blog sections link forward to lab steps; lab steps link back.
5. **Predict before run.** Every experiment asks its question before the code cell.
6. **Each section answers a question raised by the previous one.**
7. **Concepts recur by design — link back, never re-teach.** Chapter 02's cosine returns in 41;
   Chapter 02's feature scaling returns in 22 and 33.

## 6. Verification workflow — do this, it catches real errors

**Python is broken on this machine** (`/usr/bin/python3` fails with an `xcrun` error). Ruby
works. So:

1. **Verify every number in Ruby before writing it into the blog.** Compute the hand
   calculations, the table values, the training outcomes and the thresholds. This has caught
   genuine errors — a claimed stability limit of ≈1 that was really 0.4997, which the notebook
   would have visibly contradicted.
2. **Generate the notebook with a Ruby script that emits JSON**, not by hand — hand-written
   `.ipynb` JSON escaping is error-prone. Pattern:

```ruby
require 'json'
def lines(text)
  out = text.split("\n", -1).map { |l| l + "\n" }
  out.pop if out.last == "\n" && text.end_with?("\n")
  out[-1] = out[-1].chomp if out.last && out.last.end_with?("\n")
  out
end
def md(id, text); { 'cell_type' => 'markdown', 'id' => id, 'metadata' => {}, 'source' => lines(text) }; end
def code(id, text)
  { 'cell_type' => 'code', 'id' => id, 'metadata' => {},
    'execution_count' => nil, 'outputs' => [], 'source' => lines(text) }
end
# cells << md('step1-problem', <<~'MD') ... MD      (quoted heredoc: no interpolation)
notebook = { 'cells' => cells,
  'metadata' => { 'kernelspec' => { 'display_name' => 'Python 3', 'language' => 'python', 'name' => 'python3' },
                  'language_info' => { 'name' => 'python' }, 'colab' => { 'provenance' => [] } },
  'nbformat' => 4, 'nbformat_minor' => 5 }
File.write(path, JSON.pretty_generate(notebook) + "\n")
```

3. **Validate the JSON parses** and re-check every assert value independently in Ruby.
4. **Say the notebook is unexecuted.** You cannot run it. Tell the user to run it in Colab and
   commit the outputs.

### Floating-point traps that have actually bitten

- Use `np.isclose` / `np.allclose`, not `==`, for anything derived (e.g. a tax column that is
  `0.6 * market` lands on `5.3999999999999995`). Exact `==` is fine only for values that are
  exactly representable, like `(9+25+49+81)/4 == 41.0`.
- Finite-difference checks: use `h = 1e-6`, not `1e-9`. Too small and catastrophic cancellation
  swamps the result.
- Gradient-descent stability limits are $2/\lambda_{\max}$ of the loss Hessian — and **the bias
  counts as a direction**. Omitting the bias column gives the wrong threshold.

## 7. Colab requirements

- Badge in cell 1 points at this chapter's own path on `main`. Folder names contain spaces, so
  URL-encode: space → `%20`, `:` → `%3A`, `'` → `%27`.
- Runs top to bottom on a **fresh CPU runtime**, no edits, under two minutes.
- **No local files** — generate data in code or download by URL.
- Seeds set so printed numbers match the blog.
- Relative markdown links to paths with spaces must be wrapped: `[text](<blog.md>)` — bare
  spaces break the link under CommonMark.

## 8. The spine — what each chapter leaves for the next

**A chapter's closing bridge must be the next chapter's opening problem.** Since a chapter
inherits exactly what the previous one left open, one column suffices. Read row $N-1$ and
$N+1$ before writing chapter $N$.

| # | Chapter | Leaves open for the next chapter |
|---|---|---|
| 01 | What Does It Mean for a Machine to Learn? | one number cannot describe a house |
| 02 | Numbers Become Vectors | one house is a vector; how do we do *all* houses at once? |
| 03 | Matrices: The Spreadsheet of Mathematics | a matrix also *does* something to space |
| 04 | A Matrix Can Transform Space | some directions survive a transformation unchanged — which? |
| 05 | Eigenvectors: What a Transformation Leaves Alone | we can describe change in space, not *rate* of change |
| 06 | Derivatives: The Compass for Learning | a loss depends on many parameters, not one |
| 07 | Partial Derivatives, Gradients and the Chain Rule | we know the direction — how far, how often? |
| 08 | Gradient Descent | our loss was chosen by taste, not derived |
| 09 | Describing Data: Mean, Variance, Distributions | a summary is not a way to *reason* about uncertainty |
| 10 | Probability: Reasoning Under Uncertainty | how should evidence change belief? |
| 11 | Bayes' Rule | can belief-updating *derive* a loss function? |
| 12 | Maximum Likelihood: Where Loss Functions Come From | we can derive an objective — now fit real models |
| 13 | Linear Regression, Properly | a line cannot output a yes/no answer |
| 14 | Logistic Regression: The One-Neuron Network | this is one neuron — what about many classes? |
| 15 | Softmax and Cross-Entropy | all of these draw straight boundaries |
| 16 | Nearest Neighbours and the Curse of Dimensionality | memorizing everything does not generalize |
| 17 | Decision Trees | one tree overfits badly |
| 18 | Ensembles: Forests and Boosting | these dominate tabular data — so when do we need networks? |
| 19 | Support Vector Machines and Kernels | kernels are hand-chosen; can features be *learned*? |
| 20 | How Do We Know If Our Model Really Learned | low loss is not the same as a useful model |
| 21 | Evaluation Metrics | metrics assume clean data; real data is not |
| 22 | Data in the Real World | clean data still overfits |
| 23 | Overfitting and Regularization | linear models still draw straight lines |
| 24 | Meet the Smallest Neural Network | stacking layers collapses to one layer (Ch 04 said so) |
| 25 | Why Does a Neuron Need an Activation Function | which functions can a network now represent? |
| 26 | Neural Networks as Function Approximators | existence is not the same as *finding* it |
| 27 | Backpropagation | doing this scalar-by-scalar does not scale |
| 28 | Matrix Backpropagation: $dW$, $db$, $dX$ by Hand | writing gradients by hand does not scale either |
| 29 | Tensors: Numbers in Many Dimensions | we can compute — can we build the whole thing unaided? |
| 30 | Build a Tiny Neural Network From Scratch | frameworks get gradients automatically — how? |
| 31 | Automatic Differentiation | it trains, but badly and slowly |
| 32 | Optimizers: SGD, Momentum, Adam | optimizers cannot fix a badly scaled network |
| 33 | Initialization and Normalization | what else must scale with model size? |
| 34 | Scaling Rules for Training | training still fails for reasons knobs cannot explain |
| 35 | A Hacker's Guide to Deep Learning | dense layers ignore the structure of the data |
| 36 | Convolution: How a Network Learns to See | shallow convnets stop improving |
| 37 | CNN Architectures: LeNet to ResNet | why do these work — what do they assume? |
| 38 | Geometry, Invariance and Equivariance | images have space; language has order |
| 39 | When Order Matters: Sequences and Memory | long-range memory fails |
| 40 | LSTM and GRU | recurrence is sequential and cannot parallelize |
| 41 | Embeddings: Representing Meaning | one fixed vector per word ignores context |
| 42 | Attention: What Should I Look At? | attention alone is not an architecture |
| 43 | Transformers: Building With Attention | what objective trains it? |
| 44 | How a Language Model Learns to Predict Text | a trained model must now *run* |
| 45 | Inference: Decoding, Sampling, KV Cache | prediction gives generation — what else can we generate? |
| 46 | Representation Learning | can we learn representations without labels? |
| 47 | Contrastive Learning | representations describe — can a model *create*? |
| 48 | Autoencoders and the Generative Idea | generation is uncontrolled |
| 49 | Conditional Generation and Diffusion | all of this assumed grid or sequence data |
| 50 | Graph Neural Networks | why do over-parameterized models generalize at all? |
| 51 | Why Neural Networks Generalize | if bigger is better, how much bigger? |
| 52 | Scaling Laws | must we train from scratch every time? |
| 53 | Transfer Learning and Fine-Tuning | reused models meet data they never saw |
| 54 | Out-of-Distribution and Robustness | failures differ by architecture — why? |
| 55 | Architectural Bias | we must *measure* representations, not describe them |
| 56 | Measuring Representations | models match data — do they match *people*? |
| 57 | Preference Learning and Policy Optimization | a model alone is not a system |
| 58 | Retrieval, Tools and Agents | a system must run in production |
| 59 | ML Systems: Serving, Quantization and Cost | everything so far was known; research is what is not |
| 60 | From Course to Research | *the course ends where research begins* |

### Structural rules that must hold

- **Chapter 12 before any loss is used.** MSE and cross-entropy are derived there, paying off
  Chapter 01's honest "squaring is a choice."
- **Chapters 20–23 before Part VI.** A reader must be able to tell a good model from a
  memorizing one before being handed a million parameters.
- **Chapter 04's punchline decides the architecture of the whole course:** composing linear maps
  yields another linear map, so a deep stack collapses into one matrix. That is exactly the
  problem Chapters 24–25 exist to solve.

### Chapter status

Chapters 01–03 are written to this standard. Chapters marked 📄 in the repository still hold
older auto-generated material and need full rewrites, not edits. Empty slots have no folder
yet — create `Lecture NN - Title/` when you write one, and copy from [`templates/`](../../../templates/).

## 9. Definition of done

- [ ] Opens with a problem, not "in this chapter we will learn".
- [ ] No formula appears before the problem that forced it.
- [ ] Every equation is derived or explicitly motivated.
- [ ] Every symbol translated into English on first use.
- [ ] One running example carries the whole chapter.
- [ ] At least one tempting wrong idea shown failing.
- [ ] All closing sections present, exercises at five levels.
- [ ] The closing bridge matches the next chapter's inherited question exactly.
- [ ] Notebook has all 13 steps; JSON validates.
- [ ] Numbers, symbols and variable names match across both files.
- [ ] **Every numeric claim in the blog is asserted in code, and verified in Ruby first.**
- [ ] Colab badge and all links use the correct encoded path.
- [ ] The user is told the notebook is unexecuted and must be run in Colab.
