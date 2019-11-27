import json
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
from models.seed_generator import SeedGenerator
from models.cloud_conv_vae import CloudConvVae, load_model
from torch.multiprocessing import Pool
import ffmpeg
from PIL import Image
import subprocess


OUTPUT_IMAGE_WIDTH = 1920
OUTPUT_IMAGE_HEIGHT = 1080
INTERPOLATION_FRAME_COUNT = 11
BATCH_DISTANCE = 5

def ecb(ex):
    print(ex)

def process_frame_tensor(image_tensor):
    pil_image = F.to_pil_image(image_tensor)
    pil_image = pil_image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT), resample=Image.NEAREST, box=(36, 0, 648, 396))
    return pil_image.tobytes()

class Streamer(object):
    def __init__(self):
        self.cpu_count = max([1, os.cpu_count()])
        self.model_path = "C:\\code\\cloud_gen_study\\.cache\\cloudvaestorage_training-scratch_train-cloud-vae-colour-20190828-042617_model-checkpoint-250.pt"
        self.dimension_file_path = "C:\\code\\cloud_gen_study\\colour_latent_dimensions.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_key_frame = torch.randn(64, requires_grad=False, device=self.device)
        mean, std = self._get_latent_dimensions()
        self.generator = SeedGenerator(mean, std)
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
            while True:
                interpolated_frames, next_key_frame = self.generator(self.last_key_frame, delta_magnitude=BATCH_DISTANCE    , n_interpolation=INTERPOLATION_FRAME_COUNT)
                image_tensor = self.image_model.decode(interpolated_frames)            
                if mp_result:
                    mp_result.wait()
                mp_result = image_conversion_pool.map_async(process_frame_tensor, [image_tensor[i] for i in range(len(image_tensor))], callback=self._stream_frame, error_callback=ecb)
                self.last_key_frame = next_key_frame
            image_conversion_pool.terminate()
            image_conversion_pool.join()

    def _configure_streaming_process(self):
        self.stream_proc = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
            .output("rtp://127.0.0.1:1234", format="rtp", pix_fmt='rgb24', g=(INTERPOLATION_FRAME_COUNT + 1), q=0)
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def _stream_frame(self, frames):
        for frame in frames:
            self.stream_proc.stdin.write(frame)

if __name__ == "__main__":
    with torch.no_grad():
        streamer = Streamer()
        streamer.generate_batch()