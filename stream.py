import json
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
from models.seed_generator import SeedGenerator
from models.cloud_conv_vae import CloudConvVae, load_model
from torch.multiprocessing import Process, set_start_method, Queue
import ffmpeg
from PIL import Image
import subprocess
import os
from subprocess import Popen
from threading import Timer

OUTPUT_IMAGE_WIDTH = 1920
OUTPUT_IMAGE_HEIGHT = 1080
INTERPOLATION_FRAME_COUNT = 23
BATCH_DISTANCE = 10
FPS = INTERPOLATION_FRAME_COUNT + 1

def configure_streaming_process(to_file):
    if to_file:
        return (
            ffmpeg
            .input('pipe:', re=None, format='rawvideo', framerate=FPS, pix_fmt='rgb24', s='{}x{}'.format(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
            .output("video.mp4", codec="mpeg4", pix_fmt="yuv420p", q=0)
            .run_async(pipe_stdin=True))
    else:
        return (
            ffmpeg
            .input('pipe:', re=None, format='rawvideo', pix_fmt='rgb24', framerate=FPS, s='{}x{}'.format(OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))
            .output("rtp://127.0.0.1:1234", format="rtp", codec="mpeg4", pix_fmt='yuv420p', r=FPS, g=FPS, q=0, b=5000000)
            .overwrite_output()
            .run_async(pipe_stdin=True))


def process_frame_tensor(image_tensor):
    pil_image = F.to_pil_image(image_tensor)
    pil_image = pil_image.resize((OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT), resample=Image.NEAREST, box=(36, 0, 648, 396))
    return pil_image.tobytes()


def get_latent_dimensions(dimension_file_path):
    with open(dimension_file_path, "r") as  dim_file:
        dimensions = json.load(dim_file)
        means = [d["mean"] for d in dimensions]
        std = [d["std"] for d in dimensions]
        return means, std


def streaming_process(queue):
    stream_proc = configure_streaming_process(False)
    while True:
        image_tensor = queue.get()
        for i in range(len(image_tensor)):
            frame_bytes = process_frame_tensor(image_tensor[i])
            stream_proc.stdin.write(frame_bytes)


def start_player(process):
    process.start()

def generation_process(start_player=False):
    with torch.no_grad():
        script_folder = os.path.dirname(__file__)
        dimension_file_path = os.path.join(script_folder, "colour_latent_dimensions.json")
        mean, std = get_latent_dimensions(dimension_file_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        last_key_frame = torch.randn(64, requires_grad=False, device=device)
        generator = SeedGenerator(mean, std, device)
        generator.to(device)
        generator.eval()

        model_path = os.path.join(script_folder, "model-checkpoint-250.pt")
        image_model = load_model(model_path)
        image_model = image_model.to(device)
        image_model.eval()
        
        tensor_queue = Queue()
        p_streaming = Process(target=streaming_process, args=(tensor_queue,))
        p_streaming.start()

        p_mplayer = None
        player_timer = None
        if start_player:
            p_mplayer = Popen(["mplayer", "stream.sdp", "-benchmark", "-fs"], cwd=script_folder)
            player_timer = Timer(15.0, start_player, args=(p_mplayer,))
            player_timer.start()

        while True:
            interpolated_frames, next_key_frame = generator(last_key_frame, delta_magnitude=BATCH_DISTANCE, n_interpolation=INTERPOLATION_FRAME_COUNT)
            image_tensor = image_model.decode(interpolated_frames)
            image_tensor_cpu = image_tensor.cpu()
            tensor_queue.put(image_tensor_cpu)
            del image_tensor
            del interpolated_frames
            last_key_frame = next_key_frame
        p_streaming.terminate()
        p_streaming.join()


if __name__ == "__main__":
    set_start_method("spawn")
    generation_process()
    