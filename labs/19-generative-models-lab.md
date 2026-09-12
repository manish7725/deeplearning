# Lab 19 — Generate New Data

Start with a simple autoencoder.

```python
import torch
import torch.nn as nn

encoder = nn.Linear(4, 2)
decoder = nn.Linear(2, 4)

x = torch.randn(32, 4)
z = encoder(x)
reconstruction = decoder(z)
loss = ((reconstruction - x) ** 2).mean()
print(loss.item())
```

### Challenges
1. Train the autoencoder.
2. Inspect the 2D latent vectors.
3. Add noise and train a denoising autoencoder.
4. Compare reconstruction with random latent decoding.
5. Read about how VAEs and diffusion models change the generation objective.

### Resources
Welch Labs is particularly useful for visual storytelling around modern AI. 3Blue1Brown provides mathematical intuition for neural networks and modern generative ideas.