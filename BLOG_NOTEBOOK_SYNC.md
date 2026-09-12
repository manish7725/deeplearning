# Blog → Notebook synchronization

The matching Markdown blog is the source of truth for lessons 01–37. Each notebook is a self-contained interactive textbook + laboratory while the `.md` blog remains intact and canonical.

## Notebook size and runtime budgets

These are **quality budgets, not hard truncation targets**. Educational completeness takes priority, but notebooks must remain practical in GitHub, Jupyter, and Google Colab.

| Budget | Target | Hard limit / action |
|---|---:|---:|
| Notebook file size | ≤ 300 KB | > 500 KB requires optimization/review |
| Notebook cells | 50–90 | > 120 requires restructuring |
| Markdown cells | 25–55 | Avoid giant copied blog cells |
| Code cells | 15–35 | Keep one conceptual action per cell |
| Single code cell runtime | < 10 sec | > 30 sec must be optimized or explicitly justified |
| Default full-notebook runtime | < 2 min | > 5 min requires optimization/review |
| Heavy experiment runtime | < 60 sec | Must be opt-in or clearly marked |
| Training steps in default path | ≤ 2,000 | Larger runs must be optional |
| Dataset size in default path | ≤ 100,000 examples | Larger datasets must be opt-in |
| Model parameters in default path | ≤ 1,000,000 | Larger models must be optional |
| Plot generation | < 5 sec each | Avoid excessive high-resolution rendering |

## Runtime tiers

Every notebook should have three conceptual runtime tiers:

### Tier 1 — Instant lesson path

The default execution path should finish in approximately **30–120 seconds** on ordinary Google Colab CPU hardware.

It must contain the complete teaching story:

`theory → hand calculation → NumPy → PyTorch → visualization → controlled experiment → failure`

### Tier 2 — Extended experiment

Optional experiments may take up to **5 minutes**. They should be clearly labelled and separated from the default path.

### Tier 3 — Research extension

Research/reproduction experiments may exceed the normal notebook budget, but must:

- be explicitly optional
- explain expected runtime
- avoid blocking the core lesson
- use small/default settings first
- provide a scalable configuration for larger runs

## Size discipline

Do not solve size problems by deleting important theory, mathematics, examples, exercises, or experiments.

Instead:

1. split giant Markdown cells into conceptual sections;
2. keep outputs cleared before commit;
3. avoid embedding large binary outputs;
4. generate plots at reasonable resolution;
5. reuse helper functions instead of duplicating large code blocks;
6. keep large datasets outside the notebook;
7. use deterministic small examples for the default path;
8. move expensive/research-scale experiments behind explicit opt-in controls.

## Output discipline

Generated notebooks must not commit enormous execution outputs. Code cells should normally have empty `outputs` and `execution_count: null` in the repository version.

The notebook should generate visualizations when executed in Colab/Jupyter rather than storing large rendered images in Git.

## Educational priority

Never compress or remove the blog's mathematical reasoning merely to satisfy a size budget. If a lesson genuinely needs more space, prefer a well-structured notebook over an artificially short one.

The final goal remains:

**one complete, readable, runnable notebook per topic — from beginner intuition to research bridge.**
