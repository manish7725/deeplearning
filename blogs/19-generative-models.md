# Blog 19 — How Can a Machine Create Something New?

So far we have discussed models that predict labels or the next token.

But what if we want a machine to generate something new?

That is the world of **generative models**.

## 1. Learning a distribution

Imagine a box containing thousands of drawings of cats.

A generative model tries to learn enough about the patterns in those drawings that it can produce a new example that looks like it belongs to the same family.

It does not need to copy one exact training image.

It learns a mathematical description of the data distribution.

## 2. Autoencoders

An autoencoder has two main parts:

`input → encoder → compressed representation → decoder → reconstruction`

The encoder turns the input into a smaller representation.

The decoder tries to reconstruct the original.

The middle representation is sometimes called a latent representation.

## 3. Variational autoencoders

A VAE introduces probability into the latent representation.

Instead of learning only one exact point, it learns a structured latent distribution.

This makes it possible to sample new points and decode them into new examples.

## 4. GANs

A Generative Adversarial Network uses two models:

- generator — creates examples
- discriminator — tries to distinguish generated examples from real ones

They compete during training.

The generator improves by trying to fool the discriminator.

## 5. Diffusion models

A diffusion model can be understood through a two-part idea:

1. gradually add noise to data during a forward process
2. learn how to reverse that process and remove noise

Starting from noise, the learned reverse process can generate a structured sample.

## 6. The common idea

Autoencoders, GANs, and diffusion models look very different.

But they share a deep idea:

> **Learn the structure of data well enough that the model can produce new samples from that learned structure.**

Generation is not simply “copying.” It is the result of learning a mathematical model of patterns.