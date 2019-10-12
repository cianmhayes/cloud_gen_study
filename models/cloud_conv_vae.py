import torch
import torch.nn as nn
from torch.nn import functional as F
from enum import Enum

class AEMode(Enum):
    EncodeAndDecode = 1
    EncodeOnly = 2
    Decodeonly = 3

def load_model(path, mode=AEMode.EncodeAndDecode):
    checkpoint = torch.load(path, map_location="cpu")
    width = checkpoint["image_width"]
    height = checkpoint["image_height"]
    hidden = checkpoint["image_hidden"]
    latent = checkpoint["image_latent"][0] # TODO: fix tuple 
    channels = checkpoint["image_channels"]
    model = CloudConvVae(width, height, hidden, latent, channels, mode=mode)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def save_model(path, model, epoch = None, optimizer = None):
    model_to_save = model
    if isinstance(model_to_save, torch.nn.DataParallel) :
        model_to_save = model.module
    output_state = {
        "image_width": model_to_save.image_width,
        "image_height": model_to_save.image_height,
        "image_hidden": model_to_save.hidden_dimensions,
        "image_latent": model_to_save.latent_dimensions,
        "image_channels": model_to_save.image_channels,
        "model_state_dict": model_to_save.state_dict(),
        }
    if epoch:
        output_state["epoch"] = epoch
    if optimizer:
        output_state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(output_state, path)

# based on https://github.com/coolvision/vae_conv/blob/master/mvae_conv_model.py

class CloudConvVae(nn.Module):
    def __init__(self, width, height, hidden, latent, channels, mode=AEMode.EncodeAndDecode):
        super(CloudConvVae, self).__init__()

        self.mode = mode
        self.image_width_value = width
        self.image_height_value = height
        self.hidden_dimensions_value = hidden
        self.latent_dimensions_value = latent,
        self.image_channels_value = channels

        self.encode_conv = nn.Sequential(
            # (C, H, W)
            # (1, 432, 648)
            nn.Conv2d(channels, 64, 8, 4, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            # (64, 108, 162)
            nn.Conv2d(64, 128, 8, 3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            # (128, 36, 54)
            nn.Conv2d(128, 256, 4, 3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),

            # (256, 12, 18)
            nn.Conv2d(256, 512, 4, 3, padding=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),

            # Theory: (512, 4, 6), Computed: (512, 3, 5)
            nn.Conv2d(512, 1024, (4,6), bias=False)
        )
        self.encode_dense = nn.Linear(1024, hidden)
        self.encode_mu = nn.Linear(hidden, latent)
        self.encode_log_variance = nn.Linear(hidden, latent)

        self.decode_latent = nn.Linear(latent, hidden)
        self.decode_dense = nn.Linear(hidden, 1024)

        self.decode_conv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (4,6), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # (512, 4, 6)
            nn.ConvTranspose2d(512, 256, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # (256, 12, 18)
            nn.ConvTranspose2d(256, 128, 4, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (128, 36, 54)
            nn.ConvTranspose2d(128, 64, 8, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (64, 108, 162)
            nn.ConvTranspose2d(64, channels, 8, 4, bias=False),

            # (1, 432, 648)
            nn.Sigmoid()
        )
        
    @property
    def image_channels(self):
        return self.image_channels_value

    @property
    def image_width(self):
        return self.image_width_value

    @property
    def image_height(self):
        return self.image_height_value

    @property
    def hidden_dimensions(self):
        return self.hidden_dimensions_value

    @property
    def latent_dimensions(self):
        return self.latent_dimensions_value

    def reparameterize(self, mu, log_variance):
        std_dev = torch.exp(0.5*log_variance)
        epsilon = torch.randn_like(std_dev)
        return mu + std_dev*epsilon

    def encode(self, input):
        conv_encoding = self.encode_conv(input)
        hidden = F.relu(self.encode_dense(conv_encoding.view(-1, 1024)))
        return self.encode_mu(hidden), self.encode_log_variance(hidden)

    def decode(self, z):
        hidden = F.relu(self.decode_latent(z))
        dense = F.relu(self.decode_dense(hidden))
        dense_view = dense.view(-1,1024,1,1)
        decode = self.decode_conv(dense_view)
        return decode

    def forward(self, image):
        if self.mode == AEMode.EncodeAndDecode:
            mu, log_variance = self.encode(image)
            z = self.reparameterize(mu, log_variance)
            return self.decode(z), mu, log_variance, z
        elif self.mode == AEMode.EncodeOnly:
            return self.encode(image)
        elif self.mode == AEMode.Decodeonly:
            return self.decode(image)
        else:
            raise Exception("Unsupported mode")