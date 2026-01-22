import os
import torch
from torchvision.utils import save_image

@torch.no_grad()
def save_class_grid(generator, device, z_dim : int, num_classes : int, out_path : str):

    generator.eval()
    z = torch.randn(num_classes, z_dim, device = device)
    labels = torch.arange(num_classes, device = device)

    imgs = generator(z, labels)
    os.makedirs(os.path.dirname(out_path), exist_ok = True)
    save_image(imgs, out_path, nrow = 5, normalize = True)
    generator.train()