import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, num_classes : int, base_channels : int =64):
        super().__init__()
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, 28*28)

        self.net = nn.Sequential(
            nn.Conv2d(2, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(base_channels, base_channels * 2,4,2,1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Flatten(),
            nn.Linear((base_channels*2)*7*7, 1),
            nn.Sigmoid(),
        )

    def forward(self, img : torch.Tensor, labels : torch.Tensor) -> torch.Tensor:
        y = self.label_emb(labels).view(-1,1,28,28)
        x = torch.cat([img,y], dim = 1)
        return self.net(x)