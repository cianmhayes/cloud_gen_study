import argparse
import torch
from models.cloud_conv_vae import CloudConvVae, AEMode, load_model

if __name__ == "__main__":

    model = load_model("C:/Users/Cían/Downloads/model-checkpoint-70.pt")
    
    device = torch.device("cpu")

    model.mode = AEMode.EncodeOnly
    dummy_input = torch.randn(1, 1, model.image_height, model.image_width).to(device)
    torch.onnx.export(model, dummy_input, "encoder.onnx")

    model.mode = AEMode.Decodeonly
    dummy_input = torch.randn(1, 1, 1, model.latent_dimensions).to(device)
    torch.onnx.export(model, dummy_input, "decoder.onnx")