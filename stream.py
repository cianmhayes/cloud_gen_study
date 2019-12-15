import json
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
from models.seed_generator import SeedGenerator
from models.cloud_conv_vae import CloudConvVae, load_model
from torch.multiprocessing import Pool, set_start_method
import ffmpeg
from PIL import Image
import subprocess
import os

OUTPUT_IMAGE_WIDTH = 1920
OUTPUT_IMAGE_HEIGHT = 1080
INTERPOLATION_FRAME_COUNT = 23
BATCH_DISTANCE = 10
FPS = INTERPOLATION_FRAME_COUNT + 1

def ecb(ex):
    print(ex)

def process_frame_tensor(image_tensor):
    pil_image = F.to_pil_image(image_tensor)
    pil_image = pil_image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT), resample=Image.NEAREST, box=(36, 0, 648, 396))
    return pil_image.tobytes()

class Streamer(object):
    def __init__(self, to_file=False, target_frame_count=None, max_processes=1):
        self.to_file = to_file
        self.target_frame_count = target_frame_count
        self.gnerated_frame_count = 0
        self.should_run = True
        self.cpu_count = min(max_processes, max([1, os.cpu_count()]))
        script_folder = os.path.dirname(__file__) 
        self.model_path = os.path.join(script_folder, "model-checkpoint-250.pt")
        self.dimension_file_path = os.path.join(script_folder, "colour_latent_dimensions.json")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_key_frame = torch.randn(64, requires_grad=False, device=self.device)
        mean, std = self._get_latent_dimensions()
        self.generator = SeedGenerator(mean, std, self.device)
        self.generator.to(self.device)
        self.generator.eval()
        self.image_model = load_model(self.model_path)
        self.image_model = self.image_model.to(self.device)
        self.image_model.eval()
        self._configure_streaming_process()

    def _get_latent_dimensions(self):
        with open(self.dimension_file_path, "r") as  dim_file:
            dimensions = json.load(dim_file)
            means = [d["mean"] for d in dimensions]
            std = [d["std"] for d in dimensions]
            return means, std

    def generate_batch(self):
        mp_result = None
        with Pool(processes=self.cpu_count) as image_conversion_pool:
            while self.should_run:
                interpolated_frames, next_key_frame = self.generator(self.last_key_frame, delta_magnitude=BATCH_DISTANCE, n_interpolation=INTERPOLATION_FRAME_COUNT)
                image_tensor = self.image_model.decode(interpolated_frames)            
                if mp_result:
                    mp_result.wait()
                image_tensor = image_tensor.cpu()
                mp_result = image_conversion_pool.map_async(process_frame_tensor, [image_tensor[i] for i in range(len(image_tensor))], callback=self._stream_frame, error_callback=ecb)
                self.last_key_frame = next_key_frame
                self.gnerated_frame_count += FPS
                if self.target_frame_count is not None and self.gnerated_frame_count >= self.target_frame_count:
                    self.should_run = False
            image_conversion_pool.terminate()
            image_conversion_pool.join()

    def _configure_streaming_process(self):
        if self.to_file:
            self.stream_proc = (
                ffmpeg
                .input('pipe:', re=None, format='rawvideo', framerate=FPS, pix_fmt='rgb24', s='{}x{}'.format(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
                .output("video.mp4", codec="mpeg4", pix_fmt="yuv420p", q=0)
                .run_async(pipe_stdin=True)
            )
        else:
            self.stream_proc = (
                ffmpeg
                .input('pipe:', re=None, format='rawvideo', pix_fmt='rgb24', framerate=FPS, s='{}x{}'.format(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
                .output("rtp://127.0.0.1:1234", format="rtp", codec="mpeg4", pix_fmt='yuv420p', r=FPS, g=FPS, q=0, b=5000000)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

    def _stream_frame(self, frames):
        for frame in frames:
            self.stream_proc.stdin.write(frame)

if __name__ == "__main__":
    with torch.no_grad():
        set_start_method("spawn")
        streamer = Streamer()
        streamer.generate_batch()