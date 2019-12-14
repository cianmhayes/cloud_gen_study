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
from multiprocessing import Process, Queue
import time
import subprocess
from collections import deque

OUTPUT_IMAGE_WIDTH = 1920
OUTPUT_IMAGE_HEIGHT = 1080
INTERPOLATION_FRAME_COUNT = 23
BATCH_DISTANCE = 5
FPS = INTERPOLATION_FRAME_COUNT + 1

MAX_BUFFER = 2 * 60 * FPS
MAX_VAMP_BUFFER = 2 * FPS

def process_frame_tensor(image_tensor):
    pil_image = F.to_pil_image(image_tensor)
    pil_image = pil_image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT), resample=Image.NEAREST, box=(36, 0, 648, 396))
    return pil_image.tobytes()

class Streamer(object):
    def __init__(self):
        self.cpu_count = max([1, os.cpu_count()])
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

    def _get_latent_dimensions(self):
        with open(self.dimension_file_path, "r") as  dim_file:
            dimensions = json.load(dim_file)
            means = [d["mean"] for d in dimensions]
            std = [d["std"] for d in dimensions]
            return means, std

    def generate_batch(self, tensor_queue: Queue) -> None:
        interpolated_frames, next_key_frame = self.generator(self.last_key_frame, delta_magnitude=BATCH_DISTANCE    , n_interpolation=INTERPOLATION_FRAME_COUNT)
        image_tensor = self.image_model.decode(interpolated_frames)            
        image_tensor = image_tensor.cpu()
        tensor_queue.put(image_tensor)
        self.last_key_frame = next_key_frame

def generation_process(tensor_queue: Queue) -> None:
    with torch.no_grad():
        streamer = Streamer()
        while True:
            if tensor_queue.qsize() < MAX_BUFFER:
                streamer.generate_batch(tensor_queue)

def render_process(tensor_queue:Queue, frame_queue:Queue) -> None:
    while True:
        image_tensor = tensor_queue.get()
        for i in range(len(image_tensor)):
            frame_bytes = process_frame_tensor(image_tensor[i])
            frame_queue.put(frame_bytes)

def stream(q: Queue, run_mplayer = False) -> None:
    stream_proc = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
            .output("rtp://127.0.0.1:1234", format="rtp", pix_fmt='rgb24', r=FPS, g=FPS, q=0)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    player_proc = None
    if run_mplayer :
        script_folder = os.path.dirname(__file__)
        player_proc = subprocess.Popen(["mplayer", "-fs", "stream.sdp"], cwd=script_folder)
    interval = 1.0 / FPS
    next_frame_time = 0
    vamp_frames = deque()
    while True:
        if q.empty() and vamp_frames.count > 0:
            vamp_buffer = []
            for i in range(vamp_frames.count):
                vamp_buffer.append(vamp_frames[i])
            for i in range(1, vamp_frames.count):
                vamp_buffer.append(vamp_frames[vamp_buffer.count - i])
            for i in range(len(vamp_buffer)):
                while time.clock() <= next_frame_time:
                    pass
                stream_proc.stdin.write(vamp_buffer[i])
                next_frame_time = time.clock() + interval
        frame = q.get()
        vamp_frames.appendleft(frame)
        if vamp_frames.count >= MAX_VAMP_BUFFER:
            vamp_frames.pop()
        while time.clock() <= next_frame_time:
            pass
        stream_proc.stdin.write(frame)
        next_frame_time = time.clock() + interval


if __name__ == "__main__":
    tensor_queue = Queue()
    frame_queue = Queue()
    p_gen = Process(target=generation_process, args=(tensor_queue,))
    p_render = Process(target=render_process, args=(tensor_queue,frame_queue))
    p_gen.start()
    p_render.start()
    stream(frame_queue)
    p_gen.join()
    p_render.join()