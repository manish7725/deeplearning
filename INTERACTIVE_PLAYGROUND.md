# Interactive Mathematics Playground

The repository should progressively turn the most important equations into things the learner can **touch, move and observe**.

## Priority playgrounds

### 1. Derivative playground

A movable point on a curve. Show:

- secant slope;
- tangent slope;
- finite-difference approximation;
- exact derivative;
- how the approximation improves as the step becomes smaller.

### 2. Gradient-descent playground

Show a loss surface with a movable starting point. Controls:

- learning rate;
- starting point;
- number of steps;
- optimizer.

Animate the parameter update:

`θ(next) = θ(now) − η ∇L(θ)`

### 3. Neural-network playground

Show each neuron as:

`z = w·x + b`

followed by the activation. Let the learner change weights, bias and inputs while displaying every intermediate number.

### 4. Backpropagation playground

Display a computation graph and animate the chain rule from loss toward earlier parameters. Show local derivatives and accumulated gradients.

### 5. Matrix playground

Represent a matrix as a geometric transformation. Animate:

- scaling;
- rotation;
- shear;
- projection;
- composition.

### 6. Convolution playground

Move a kernel across an image and expose the multiplication-and-sum calculation for every output pixel.

### 7. Attention playground

For a tiny sentence, display Q, K and V matrices, attention scores, softmax probabilities and the weighted sum. Let the learner change one token embedding.

### 8. Embedding playground

Plot a tiny learned embedding space. Show distances and cosine similarity while points move.

### 9. Optimization comparison

Run the same objective with SGD, momentum and Adam. Display trajectories and update magnitudes.

### 10. Scaling-law playground

Generate or load measured training runs and fit:

`loss ≈ A · compute^(-α) + C`

Show residuals and confidence intervals rather than only a fitted line.

## Design rule

Every playground must have a **static fallback**. Interactive graphics explain behaviour; the blog and notebook contain the mathematical truth.

## Implementation direction

Prefer small self-contained JavaScript visualizations or notebook widgets when they materially improve understanding. Keep dependencies minimal. A learner should be able to inspect the source and understand how the visualization corresponds to the equation.
