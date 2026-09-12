# Blog 09 — Gradient Descent: Teaching a Model to Improve

We know what the derivative tells us.

Now we can use it to train a model.

The algorithm is called **gradient descent**.

## 1. One parameter first

Suppose:

`L(w) = (w - 5)²`

We want `w` to become 5 because that gives the smallest loss.

The derivative is:

`dL/dw = 2(w - 5)`

Start with:

`w = 1`

Then:

`gradient = 2(1-5) = -8`

## 2. Take a step

The update rule is:

`w_new = w_old - η × gradient`

Here `η` is the **learning rate**.

Suppose:

`η = 0.1`

Then:

`w_new = 1 - 0.1(-8)`

`= 1.8`

We moved toward 5.

## 3. Repeat

At `w = 1.8`:

`gradient = 2(1.8-5) = -6.4`

Update:

`w = 1.8 - 0.1(-6.4)`

`= 2.44`

Again we moved closer.

Eventually, repeated updates bring us near the minimum.

## 4. Why “descent”?

Imagine the loss function as a landscape.

The gradient points toward increasing loss.

So we move in the opposite direction:

`-gradient`

That is why the algorithm is called gradient **descent**.

## 5. Learning rate matters

If the learning rate is tiny, learning can be very slow.

If it is enormous, the model may jump over the minimum and become unstable.

So the learning rate is like the size of your walking step.

Small steps are careful but slow.
Large steps are fast but can make you stumble.

## 6. Many parameters

For a neural network, we do not have one parameter `w`.

We have a parameter vector or matrix `W`.

The idea stays the same:

`W_new = W_old - η∇L(W)`

The symbol `∇` means gradient.

The formula looks advanced, but its meaning is simple:

**change every parameter in the direction that reduces the loss.**

That is one of the central equations of deep learning.