# Blog 08 — Derivatives: A Compass for Learning

Suppose you are walking down a hill in fog.

You cannot see the whole landscape, but you can feel whether the ground slopes upward or downward beneath your feet.

A derivative gives us a similar idea for mathematics.

## 1. A function changes

Take:

`y = x²`

If `x = 2`, then:

`y = 4`

If `x = 3`, then:

`y = 9`

The output changes when the input changes.

The derivative tells us how quickly it changes.

For:

`y = x²`

we have:

`dy/dx = 2x`

At `x = 3`:

`dy/dx = 6`

That means a small movement in `x` around 3 causes the output to change roughly six times as much.

## 2. Derivatives tell us direction

Suppose our loss is a function of a parameter:

`L(w) = (w - 5)²`

The best value is `w = 5`, because the loss becomes zero.

Different values give larger loss.

The derivative is:

`dL/dw = 2(w - 5)`

At `w = 2`:

`dL/dw = 2(2-5) = -6`

The negative sign tells us something important: increasing `w` will reduce the loss when we are at `w=2`.

## 3. Think of the derivative as a signpost

Positive derivative → moving right increases the function.

Negative derivative → moving right decreases the function.

Zero derivative → locally, the function is flat.

This is exactly the information an optimization algorithm needs.

## 4. From one parameter to thousands

A neural network may have millions or billions of parameters.

For each parameter, we want to know:

“How would the loss change if I changed this number slightly?”

The answer for every parameter together forms the **gradient**.

The gradient is therefore a giant collection of derivatives.

## 5. The deep-learning connection

Training a neural network means reducing its loss.

The gradient tells us how the loss changes with respect to the parameters.

So derivatives become our mathematical compass.

> **Calculus turns “make the model better” into a precise numerical direction for changing its parameters.**

Next we will use that compass to actually walk downhill.