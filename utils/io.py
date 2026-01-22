import os
import torch

def ensure_dir(path : str) -> None:
    os.makedirs(path, exist_ok = True)

def save_checkpoint(path : str, generator, discriminator, opt_g, opt_d, epoch : int) -> None:
    payload = {
        "epoch" : epoch,
        "generator" : generator.state_dict(),
        "discriminator" : discriminator.state_dict(),
        "opt_g" : opt_g.state_dict(),
        "opt_d" : opt_d.state_dict()
    }
    torch.save(payload, path)


def load_checkpoint(path : str, generator, discriminator, opt_g = None, opt_d = None, map_location = "cpu"):
    ckpt = torch.load(path, map_location = map_location)
    generator.load_state_dict(ckpt["generator"])
    if discriminator is not None and "discriminator" in ckpt:
        discriminator.load_state_dict(ckpt["discriminator"])
    if opt_g is not None and "opt_g" in ckpt:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and "opt_d" in ckpt:
        opt_d.load_state_dict(ckpt["opt_d"])
    return ckpt.get("epoch", 0)