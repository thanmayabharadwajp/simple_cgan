import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim : int, num_classes : int, base_channels : int = 128):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.fc = nn.Sequential(
            nn.Linear(z_dim + num_classes, base_channels * 7 * 7),
            nn.BatchNorm1d(base_channels*7*7),
            nn.ReLU(True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2D(base_channels, base_channels // 2,4,2,1),
            nn.BatchNorm2d (base_channels // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_channels // 2,1,4,2,1),
            nn.Tanh(), # output in [-1,1]
        )

    def forward(self, z : torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y = self.label_emb(labels)
        x = torch.cat([z,y], dim = 1) # concatenation

        x = self.fc(x)
        x = x.view(x.size(0), -1,7,7)
        img = self.deconv(x)
        return img