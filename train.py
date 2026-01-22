import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import CFG
from models import Generator, Discriminator
from utils.seed import set_seed
from utils.data import get_fashion_mnist_loader
from utils.io import ensure_dir, save_checkpoint, load_checkpoint
from utils.viz import save_class_grid

def get_device(force_cpu : bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type = int, default = CFG.epochs)
    p.add_argument("--batch_size", type = int, default = CFG.batch_size)
    p.add_argument("--lr", type = float, default = CFG.lr)
    p.add_argument("--seed", type = int, default = CFG.seed)
    p.add_argument("--force_cpu", action = "store_true", help = "Force CPU even if CUDA is available")
    p.add_argument("--resume", type = str, default = "", help = "Path to checkpoint to resume from.")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device (force_cpu = args.force_cpu)
    print(f"[Info] Using device : {device}")

    ensure_dir(CFG.samples_dir)
    ensure_dir(CFG.checkpoints_dir)

    loader = get_fashion_mnist_loader(CFG.data_root, args.batch_size, CFG.num_workers)

    G = Generator(CFG.z_dim, CFG.num_classes, CFG.g_base_channels).to(device)
    D = Discriminator(CFG.num_classes, CFG.d_base_channels).to(device)

    opt_G = optim.Adam(G.parameters(), lr = 2e-4, betas = CFG.betas)
    opt_D = optim.Adam(D.parameters(), lr = 1e-4, betas = CFG.betas)
    criterion = nn.BCELoss()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, G, D, opt_G, opt_D,
            map_location = "cpu" if device.type == "cpu" else device
        )
        print(f"[Info] Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()

        pbar = tqdm(loader, desc = f"Epoch {epoch+1}/{args.epochs}")
        last_d_loss, last_g_loss = None, None

        for real_imgs, labels in pbar:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            bsz = real_imgs.size(0)

            real_target = torch.full((bsz, 1), 0.9, device = device)
            fake_target = torch.zeros(bsz, 1, device = device)

            # 1. Train Discriminator

            z = torch.randn(bsz, CFG.z_dim, device = device)
            fake_imgs = G(z, labels)

            d_real = D(real_imgs, labels)
            d_fake = D(fake_imgs.detach(), labels)
            d_loss = criterion(d_real, real_target) + criterion(d_fake, fake_target)

            opt_D.step()

            # 2. Train generator
            
            d_fake_for_g = D(fake_imgs, labels)
            g_loss = criterion(d_fake_for_g, real_target)

            opt_G.zero_grad(set_to_none = True)
            g_loss.backward()
            opt_G.step()

            last_d_loss = d_loss.item()
            last_g_loss = g_loss.item()
            pbar.set_postfix(D_loss=f"{last_d_loss:.4f}", G_loss=f"{last_g_loss:.4f}")

        # Save sample grid (one per class)
        if (epoch + 1) % CFG.sample_epochs == 0:
            out_img = os.path.join(CFG.samples_dir, f"epoch_{epoch+1:03d}.png")
            save_class_grid(G, device, CFG.z_dim, CFG.num_classes, out_img)

        # Save checkpoint
        if (epoch + 1) % CFG.save_epochs == 0:
            ckpt_path = os.path.join(CFG.checkpoints_dir, f"cgan_epoch_{epoch+1:03d}.pt")
            save_checkpoint(ckpt_path, G, D, opt_G, opt_D, epoch + 1)

        print(f"[Epoch {epoch+1}] D_loss={last_d_loss:.4f}  G_loss={last_g_loss:.4f}")

    print("[Done] Training complete.")
    print(f"Samples saved to: {CFG.samples_dir}/")
    print(f"Checkpoints saved to: {CFG.checkpoints_dir}/")

if __name__ == "__main__":
    main()