# Blog 20 — Build a Tiny Neural Network From Scratch

We have traveled a long road.

We started with numbers.
Then vectors.
Then matrices.
Then neurons.
Then loss, derivatives, gradient descent, backpropagation, tensors, CNNs, sequences, embeddings, attention, and Transformers.

Now let's put the pieces together.

## 1. Our tiny network

Suppose we have two inputs:

`x = [2, 3]`

We use two weights:

`w = [0.5, 1.0]`

and a bias:

`b = 0.5`

The neuron calculates:

`z = 2×0.5 + 3×1.0 + 0.5`

`z = 4.5`

Suppose we use ReLU:

`a = max(0, 4.5) = 4.5`

We have just performed a forward pass.

## 2. Compare with the answer

Suppose the correct answer is:

`y = 5`

Use squared error:

`L = (5 - 4.5)²`

`L = 0.25`

The model is not perfect, but it is reasonably close.

## 3. Find the direction to improve

Backpropagation calculates how the loss changes with respect to the parameters.

For every weight and bias, we obtain a gradient.

Conceptually:

`gradient = how much this parameter affects the loss`

## 4. Update the parameters

Choose a learning rate, say:

`η = 0.1`

Then update each parameter:

`parameter_new = parameter_old - η × gradient`

The parameters move slightly in a direction expected to reduce the loss.

## 5. Repeat many times

The complete algorithm becomes:

```text
for each training step:
    1. read the input
    2. calculate the prediction
    3. calculate the loss
    4. calculate gradients
    5. update parameters
```

After many steps, the model may become much better at the task.

## 6. The PyTorch version

```python
import torch

x = torch.tensor([[2.0, 3.0]])
y = torch.tensor([[5.0]])

model = torch.nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(100):
    prediction = model(x)
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(x))
```

There is a lot hidden inside these few lines.

`Linear` contains parameters.

`prediction = model(x)` performs the forward computation.

`loss_fn` measures the mistake.

`loss.backward()` performs automatic differentiation and calculates gradients.

`optimizer.step()` updates the parameters.

## 7. What you should remember

A modern deep-learning system may contain billions of parameters and enormous datasets.

Yet the central loop is still remarkably close to our tiny example:

**represent information → transform it → predict → measure error → calculate gradients → update parameters → repeat**

That is the mathematical heart of deep learning.

## Final challenge

Do not stop at understanding these ideas.

Implement them.

First with ordinary Python.
Then with NumPy.
Then with PyTorch.
Then inspect every tensor shape and every gradient.

A strong programmer does not merely memorize the formula.

A strong mathematician asks why the formula works.

A strong data scientist asks whether the model works on unseen data.

And a strong learner combines all three questions.

> **Deep learning becomes much less mysterious when you keep reducing every impressive system back to first principles: numbers, functions, matrices, calculus, data, and computation.**

This is only the beginning.