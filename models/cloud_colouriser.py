import torch
import torch.nn as nn
from torch.nn import functional as F

class CloudColouriser(nn.Module):
    def __init__(self):
        super(CloudColouriser, self).__init__()

        self.patch_extraction = nn.Conv2d(1, 64, 17, padding=8)
        self.non_linear = nn.Conv2d(64, 32, 1)
        self.reconstruction = nn.Conv2d(32, 3, 5, padding=2)

    def forward(self, input):
        patches = F.leaky_relu(self.patch_extraction(input))
        mapped = F.leaky_relu(self.non_linear(patches))
        reconstructed = self.reconstruction(mapped)
        return reconstructed