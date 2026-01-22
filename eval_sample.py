# eval_sample.py
import argparse
import os
import torch

from config import CFG
from models import Generator
from utils.io import load_checkpoint, ensure_dir
from utils.viz import save_class_grid

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--out", type=str, default="samples/eval_grid.png")
    p.add_argument("--force_cpu", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu" if args.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Info] Using device: {device}")

    G = Generator(CFG.z_dim, CFG.num_classes, CFG.g_base_channels).to(device)

    # Load only generator weights from checkpoint
    load_checkpoint(args.ckpt, G, discriminator=None, opt_g=None, opt_d=None,
                   map_location="cpu" if device.type == "cpu" else device)

    ensure_dir(os.path.dirname(args.out))
    save_class_grid(G, device, CFG.z_dim, CFG.num_classes, args.out)
    print(f"[Done] Saved: {args.out}")

if __name__ == "__main__":
    main()
