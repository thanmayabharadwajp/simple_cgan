# cGAN Benchmark (Fashion-MNIST)

A clean, baseline conditional GAN (cGAN) for Fashion-MNIST, designed for benchmarking.
- Generator: (z, label) -> image
- Discriminator: (image, label) -> real/fake

## Setup
```bash
pip install -r requirements.txt
```

Conditional GAN (cGAN) Benchmark – Fashion-MNIST

This repository implements a clean, minimal Conditional GAN (cGAN) in PyTorch, trained on the Fashion-MNIST dataset.
The project was developed CPU-first for correctness, then migrated to GPU for performance, with a strong focus on training stability, reproducibility, and debugging correctness.

This model serves as a benchmark baseline for future generative experiments.

📌 Project Goals

Build a from-scratch cGAN (no boilerplate code)

Understand GAN training dynamics deeply (loss behavior, instability)

Debug common GAN failure modes (loss saturation, blur, imbalance)

Produce a reproducible benchmark that can be extended later

🧠 Model Overview
Generator

Input:

Noise vector z ∈ ℝ¹⁰⁰

Class label y ∈ {0,…,9}

Conditioning:

Label is embedded and concatenated with noise

Architecture:

Fully connected → reshape → transposed convolutions

Batch Normalization for stability

tanh output (images in [-1, 1])

Discriminator

Input:

Image x

Class label y

Conditioning:

Label embedded into a spatial map and concatenated with image

Architecture:

Convolutional layers

Batch Normalization

Sigmoid output for real/fake probability

📂 Repository Structure
simple_cgan/
├── models/
│   ├── generator.py        # Conditional Generator
│   └── discriminator.py    # Conditional Discriminator
│
├── utils/
│   ├── data.py             # Dataset & DataLoader
│   ├── seed.py             # Reproducibility utilities
│   ├── viz.py              # Sample visualization
│   └── io.py               # Checkpoint saving/loading
│
├── config.py               # Central configuration
├── train.py                # Training loop
├── README.md               # This file
│
├── samples/                # Selected generated samples
│   ├── epoch_001.png
│   ├── epoch_005.png
│   └── epoch_010.png
│
├── checkpoints/
│   └── final.pt            # Final trained checkpoint

⚙️ Training Details
Dataset

Fashion-MNIST

60,000 training images

10 classes

Images normalized to [-1, 1]

Hyperparameters
Parameter	Value
Latent dim (z_dim)	100
Batch size	32 (CPU) / 128+ (GPU)
Generator LR	2e-4
Discriminator LR	1e-4
Adam betas	(0.5, 0.999)
Epochs	30
🧪 Training Stability Techniques Used

To address known GAN instability issues, the following techniques were applied:

Label smoothing

Real labels set to 0.9 instead of 1.0

Asymmetric learning rates

Discriminator trained slower than Generator

Batch Normalization

Used extensively in Generator

Careful gradient handling

Explicit zero_grad → backward → step order

Early debugging via visual inspection

Images used as primary metric, not losses

📉 Understanding the Loss Behavior

GAN losses are not directly interpretable like supervised learning losses.

Observed behavior:

Early epochs:

Discriminator loss very high

Generator loss very low

This is expected and does not indicate failure

Image quality, not loss values, was used to judge training progress

🖼️ Sample Outputs

The samples/ directory contains representative outputs at different epochs:

Epoch 1–2: Random noise

Epoch 3–5: Blurry silhouettes

Epoch 6–10: Clear class-dependent shapes

These samples demonstrate correct conditional learning behavior.

🚀 Running the Code
CPU (development / debugging)
python train.py --force_cpu

GPU (recommended)

Ensure CUDA-enabled PyTorch is installed, then:

python train.py


The script automatically detects CUDA and switches to GPU.

🔁 Reproducibility

Fixed random seeds

Deterministic cuDNN settings (when applicable)

Centralized configuration in config.py

📦 Checkpoints & Version Control

Only selected samples and one final checkpoint are committed

Intermediate checkpoints are ignored to keep the repository clean

This follows standard research-oriented GitHub practices

🧭 Current Status

✔ Stable training
✔ Correct conditional behavior
✔ Clean benchmark baseline
✔ Ready for extensions

🔮 Possible Extensions

Switch to WGAN-GP for improved stability

Quantitative metrics (FID, IS, conditional accuracy)

Higher-resolution datasets

Architectural experiments

Comparison with diffusion models