# Learning Contract

A repository lesson is considered complete only when the learner can move between intuition, mathematics, code and evidence.

## Blog requirements

Every blog should contain, where applicable:

1. Why the problem exists
2. A Class-8-friendly intuition
3. A concrete numerical example
4. Precise mathematical notation
5. A hand calculation
6. Matrix/tensor formulation
7. A visual explanation
8. Previous → current → next bridge
9. Common misconception
10. Failure mode
11. Exercises
12. Link to the matching notebook
13. For advanced lessons: papers and research questions

## Notebook requirements

Every matching notebook should contain:

1. Learning objective
2. Link/reference to the blog
3. Imports explained for beginners
4. Tiny hand-calculated example
5. NumPy implementation
6. PyTorch implementation where appropriate
7. Assertions for important equations/shapes
8. Visualization
9. At least one experiment with a changed variable
10. Interpretation of the result
11. Failure-mode experiment
12. Exercises/challenges
13. Mini-project for major milestones
14. Reproducibility information

## Interactive mathematics

When an equation has a meaningful geometric or optimization interpretation, prefer an interactive widget or animation. The static derivation must remain readable without JavaScript.

## Research requirements

Research-level lessons add:

- a paper or primary source;
- baseline definition;
- reproducible setup;
- controlled variables;
- ablation plan;
- multiple seeds when stochasticity matters;
- uncertainty or confidence intervals when appropriate;
- failure analysis;
- explicit distinction between observation and interpretation.

## Accessibility rule

A young learner should be able to understand the *idea* without knowing calculus. Advanced sections may introduce the calculus needed to make the idea exact.

## Quality rule

Never hide a difficult step behind a library call when the educational goal is to understand that step. Use libraries after exposing the underlying operation.
