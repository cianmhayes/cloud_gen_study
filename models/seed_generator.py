import torch
import torch.nn as nn
from torch.nn import functional as f
import math

Z_95 = 1.96
Z_98 = 2.326
Z_99 = 2.576
Z_999 = 3.29

class SeedGenerator(nn.Module):
    def __init__(self, mean, std, device):
        super().__init__()
        self.dim_mean = torch.as_tensor(mean).to(device)
        self.dim_std = torch.as_tensor(std).to(device)

    def forward(self, last_tensor, delta_magnitude=10, n_key_frames=1, n_interpolation=23):
        # generate directional tensor
        target_tensor = torch.randn_like(last_tensor)
        magnitude = math.sqrt(float(torch.sum(torch.pow(target_tensor, 2))))
        target_tensor = torch.mul(target_tensor, delta_magnitude / magnitude)

        # add it to the last tensor and clamp based on standard normal
        target_tensor = last_tensor + target_tensor
        target_tensor = target_tensor.clamp_(-1 * Z_99, Z_99)

        last_z_tensor = torch.add(torch.mul(last_tensor, self.dim_std), self.dim_mean)
        target_z_tensor = torch.add(torch.mul(target_tensor, self.dim_std), self.dim_mean)

        # interpolate
        interpolated_frames = []
        for i in range(1, n_interpolation):
            interpolated_frame = torch.lerp(last_z_tensor, target_z_tensor, (i / n_interpolation))
            interpolated_frames.append(interpolated_frame)
        interpolated_frames.append(target_z_tensor)
        interpolated_frames = torch.stack(interpolated_frames)
        return interpolated_frames, target_tensor
