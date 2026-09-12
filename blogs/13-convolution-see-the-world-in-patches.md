# Blog 13 — How a Neural Network Learns to See: Convolution

A photograph can contain millions of pixel values.

Giving all those numbers to a giant layer is possible, but it ignores an important fact about images: nearby pixels are related.

Convolution gives neural networks a clever way to use that structure.

## 1. A tiny image

Imagine this grayscale image:

```text
1 1 1 0 0
1 1 1 0 0
1 1 1 0 0
0 0 0 1 1
0 0 0 1 1
```

We can look at small regions instead of the whole image at once.

## 2. A small filter

Consider a 3×3 filter:

```text
1 0 1
0 1 0
1 0 1
```

The filter moves across the image.

At every position, we multiply corresponding numbers and add them.

That produces one output number.

Repeating this creates a **feature map**.

## 3. What can a filter discover?

Different filters can respond strongly to different patterns:

- vertical edges
- horizontal edges
- corners
- textures
- simple shapes

Early layers often detect simple structures. Deeper layers can combine them into more complex patterns.

A network might progress conceptually like:

`edges → shapes → parts → objects`

## 4. Why sharing matters

The same filter is reused across the image.

That means the network does not need a completely different detector for an edge in the top-left and an edge in the bottom-right.

One learned filter can search everywhere.

This is called **weight sharing**.

## 5. Convolution is still multiplication and addition

The operation may look sophisticated, but underneath it is the familiar pattern:

`multiply → add → move → repeat`

That is one of the recurring themes of deep learning.

Simple mathematics becomes powerful when repeated systematically.

## 6. The bigger idea

A convolutional neural network, or CNN, uses local patterns and shared filters to build useful visual representations.

> **A CNN learns what visual patterns are useful instead of requiring a human to write every visual rule by hand.**