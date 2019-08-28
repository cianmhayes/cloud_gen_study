import torch
import torch.nn as nn
from torch.nn import functional as F

class CloudSimpleVae(nn.Module):
    def __init__(self, width, height, hidden, latent):
        super(CloudSimpleVae, self).__init__()

        self.image_width = width
        self.image_height = height
        self.hidden_dimension = hidden
        self.latent_dimensions = latent
        self.initial_encode = nn.Linear(width * height, hidden)
        self.encode_mu = nn.Linear(hidden, latent)
        self.encode_log_variance = nn.Linear(hidden, latent)
        self.decode_latent = nn.Linear(latent, hidden)
        self.output = nn.Linear(hidden, width * height)
        
    def reparameterize(self, mu, log_variance):
        std_dev = torch.exp(0.5*log_variance)
        epsilon = torch.randn_like(std_dev)
        return mu + epsilon*std_dev

    def encode(self, input):
        hidden = F.relu(self.initial_encode(input))
        return self.encode_mu(hidden), self.encode_log_variance(hidden)

    def decode(self, z):
        hidden = F.relu(self.decode_latent(z))
        return F.sigmoid(self.output(hidden))

    def forward(self, image):
        mu, log_variance = self.encode(image)
        z = self.reparameterize(mu, log_variance)
        return self.decode(z), mu, log_variance, z