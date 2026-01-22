from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_fashion_mnist_loader(data_root : str, batch_size : int, num_workers : int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # map to [-1,1]
    ])

    dataset = datasets.FashionMNIST(
        root = data_root,
        train = True,
        download = True,
        transform = transform
    )

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,
        drop_last = True
    )
    return loader