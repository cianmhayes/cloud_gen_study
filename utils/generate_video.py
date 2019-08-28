import ffmpeg
import os
import sys
import argparse
import uuid
from datetime import datetime
from models.cloud_conv_vae import load_model, CloudConvVae
from azure.storage.blob import BlockBlobService
from azure.storage.file.fileservice import FileService
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torch

class VideoGenerator(object):
    def __init__(self, initial_image, frame_count=240):
        self.start_date = datetime.utcnow()
        self.frame_count = frame_count
        self.blob_service = BlockBlobService(account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"], account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"])
        self.file_service = FileService(account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"], account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"])
        self.blob_service.get_blob_to_path("training-data-w648-h432-201907131642", initial_image, initial_image)
        #self.file_service.get_file_to_path("training-scratch", "train-cloud-vae-20190713-181113", "model-checkpoint-200.pt", "model-checkpoint-200.pt")
        self.model = load_model("model-checkpoint-200.pt")
        self.im = ToTensor()(Image.open(initial_image)).view(-1, 1, 432, 648)
        os.remove(initial_image)
        _, self.initial_image_mu, self.initial_image_log_var, self.initial_image_z = self.model(self.im)

    def _draw(self):
        std_dev = torch.exp(0.5* self.initial_image_log_var)
        epsilon = torch.randn_like(std_dev)
        return self.initial_image_mu + epsilon*std_dev

    def generate_video_by_repeated_draw(self):
        folder = str(uuid.uuid4())
        os.makedirs(folder)
        for i in range(self.frame_count):
            new_frame = self.model.decode(self._draw())
            save_image(new_frame, "{0}/{1:04d}.jpg".format(folder, i))
        self._compile_video(folder, "repeated_draw")
            
    def generate_video_by_uniform_noise(self, noise_scale = 0.1):
        folder = str(uuid.uuid4())
        os.makedirs(folder)
        for i in range(self.frame_count):
            new_frame = self.model.decode(self.initial_image_z + (torch.randn_like(self.initial_image_z) * noise_scale))
            save_image(new_frame, "{0}/{1:04d}.jpg".format(folder, i))
        self._compile_video(folder, "uniform_noise_s{}".format(noise_scale))

    def generate_video_by_cumulative_uniform_noise(self, noise_scale = 0.1):
        last_z = self.initial_image_z
        folder = str(uuid.uuid4())
        os.makedirs(folder)
        for i in range(self.frame_count):
            last_z += (torch.randn_like(self.initial_image_z) * noise_scale)
            new_frame = self.model.decode(last_z)
            save_image(new_frame, "{0}/{1:04d}.jpg".format(folder, i))
        self._compile_video(folder, "cumulative_uniform_noise_s{}".format(noise_scale))

    def _compile_video(self, folder, video_root):
        video_name = "{}_{:%Y%m%d%H%M}.mp4".format(video_root, self.start_date)
        video_proc = (
            ffmpeg
            .input(folder + "/%04d.jpg", pattern_type="sequence", framerate=24)
            .output(video_name)
            .run()
            )
        self.blob_service.create_blob_from_path("videos", video_name, video_name)


if __name__ == "__main__":
    g = VideoGenerator("0fe13b8d-195a-4c49-a98b-cc009140f256.51c32088-d662-47fc-8a8e-2e84b759aeae.png", frame_count=1440)
    g.generate_video_by_cumulative_uniform_noise(noise_scale=0.001)
    g.generate_video_by_cumulative_uniform_noise(noise_scale=0.01)
    g.generate_video_by_cumulative_uniform_noise(noise_scale=0.1)