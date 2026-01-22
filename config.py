# config.py

from dataclasses import dataclass

@dataclass

class Config:
    # data
    dataset : str = "FashionMNIST"
    data_root : str = "data"
    img_size : int = 28
    img_channels : int = 128
    img_channels : int = 1
    num_classes : 10

    # model
    z_dim: int = 100
    g_base_channels : int = 128
    d_base_channels : int = 64

    # training
    epochs : int = 30 # can be higher for GPU
    batch_size : int = 128
    learning_rate : float = 2e-4
    betas : tuple = (0.5, 0.999)
    num_workers : int = 2

    # outputs
    seed : int = 42
    sample_epochs : int = 1
    save_epochs : int = 5

    # directories
    samples_dir: str = "samples"
    checkpoints_dir : str = "checkpoints"

CFG = Config()