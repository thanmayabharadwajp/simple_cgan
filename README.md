# cGAN Benchmark (Fashion-MNIST) — CPU first, GPU later

A clean, baseline conditional GAN (cGAN) for Fashion-MNIST, designed for benchmarking.
- Generator: (z, label) -> image
- Discriminator: (image, label) -> real/fake

## Setup
```bash
pip install -r requirements.txt
