import argparse
import os
from azure.storage.file import FileService
from models.cloud_conv_vae import CloudConvVae, load_model
from utils.image_normalizer import fix_aspect_ratio, resize_preserve_aspect, ImageNormalizer
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
from torchvision.utils import save_image
import json
import math
import ffmpeg
import scipy.stats as stats
from sklearn.manifold import TSNE

default_cache_folder = os.path.join(os.path.realpath(os.path.dirname(__file__)), ".cache")

def get_model(account_name, account_key, share_name, directory_name, file_name, cache_folder = default_cache_folder):
    cache_name = "_".join([account_name, share_name, directory_name, file_name])
    cache_file_path = os.path.join(cache_folder, cache_name)
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    if not os.path.exists(cache_file_path):
        fs = FileService(account_name=account_name, account_key=account_key)
        fs.get_file_to_path(share_name, directory_name, file_name, cache_file_path)
    return load_model(cache_file_path)

def get_image(image_path, model):
    im = Image.open(image_path)
    im = resize_preserve_aspect(im, model.image_width, model.image_height)
    im = fix_aspect_ratio(im, model.image_width, model.image_height)
    if not (im.size[0] == model.image_width and im.size[1] == model.image_height) :
        im = im.resize((model.image_width, model.image_height), Image.ANTIALIAS)
    if model.image_channels == 3:
        im = im.convert("RGB")
    elif model.image_channels == 1:
        im = im.convert("L")
    return im

def get_images(source_dir, model):
    for (dirpath, _, filenames) in os.walk(source_dir):
        for filename in filenames:
            source = os.path.join(dirpath, filename)
            im = get_image(source, model)
            yield im, filename


def generate_image_from_z(model, z_tensor, file_path):
    decoded_image = model.decode(z_tensor)
    save_image( decoded_image, file_path)


def interpolate(key_frames, n_interpolation_frames):
    result = []
    for  i in range(len(key_frames) - 1):
        result.append(key_frames[i])
        delta = (key_frames[i+1] - key_frames[i]) / (n_interpolation_frames + 1)
        for j in range(1, n_interpolation_frames+1):
            result.append(key_frames[i] + (delta * j))
    return result

def normalize_z_values(frames):
    with open("C:/code/cloud_gen_study/mono_latent_dimensions.json", "r") as param_file:
        ld_params = json.load(param_file)
    min_values = [p["intervals"][2]["lower"] for p in ld_params]
    max_values = [p["intervals"][2]["upper"] for p in ld_params]
    for frame in frames:
        norm_values = []
        for i in range(len(min_values)):
            norm_values.append(float((frame[i] - min_values[i]) / (max_values[i] - min_values[i])))
        yield norm_values

def wander3(starting_frame, n_frames, delta_magnitude):
    with open("C:/code/cloud_gen_study/colour_latent_dimensions.json", "r") as param_file:
        ld_params = json.load(param_file)
    #min_values = [p["intervals"][2]["lower"] for p in ld_params]
    #max_values = [p["intervals"][2]["upper"] for p in ld_params]
    min_values = [(p["mean"] + (p["std"] * -3)) for p in ld_params]
    max_values = [(p["mean"] + (p["std"] * 3)) for p in ld_params]
    result = [starting_frame.clone()]
    for i in range(int(n_frames)):
        next_direction = torch.randn_like(starting_frame)
        magnitude = math.sqrt(sum([float(next_direction[d])**2 for d in range(len(next_direction))]))
        next_direction = torch.div(next_direction, magnitude)
        next_direction = torch.mul(next_direction, delta_magnitude)
        next_frame = result[i] + next_direction
        for i in range(len(min_values)):
            if next_frame[i] < min_values[i]:
                next_frame[i] = min_values[i]
            elif next_frame[i] > max_values[i]:
                next_frame[i] = max_values[i]
        result.append(next_frame)
    return result

def wander2(starting_frame, n_frames, new_frame_weighting):
    with open("C:/code/cloud_gen_study/colour_latent_dimensions.json", "r") as param_file:
        ld_params = json.load(param_file)
    mean_tensor = torch.tensor([p["mean"] for p in ld_params])
    std_tensor = torch.tensor([p["std"] for p in ld_params])
    result = [starting_frame.clone()]
    for i in range(int(n_frames)):
        next_destination = torch.normal(mean_tensor, std_tensor)
        result.append((result[i] * (1.0 - new_frame_weighting)) + (next_destination * new_frame_weighting))
    return result


def wander(starting_frame, n_frames, delta_magnitude):
    result = [starting_frame.clone()]
    for i in range(int(n_frames)):
        next_direction = torch.randn_like(starting_frame)
        magnitude = math.sqrt(sum([float(next_direction[d])**2 for d in range(len(next_direction))]))
        next_direction = torch.div(next_direction, magnitude)
        next_direction = torch.mul(next_direction, delta_magnitude)
        result.append(result[i] + next_direction)
    return result


def evaluate_model(model, im):
    image_tensor = ToTensor()(im).view(-1, model.image_channels, model.image_height, model.image_width)
    transcoded_image_tensor, _, _, z = model(image_tensor)
    return transcoded_image_tensor, z

def transcode_images(model, source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    encodings_list = []
    for im, filename in get_images(source_dir, model):
        try:
            dest_file_name = os.path.join(dest_dir, filename)
            transcoded_image_tensor, z = evaluate_model(model, im)
            z_list = z.flatten().tolist()
            encodings_list.append({"file_name": filename, "encoding": z_list})
            save_image(transcoded_image_tensor, dest_file_name)
        except Exception as e:
            print("Failed to transcode {} due to {}".format(filename, str(e)))
    tsne_proj = TSNE(random_state=20190914).fit_transform([e["encoding"] for e in encodings_list])
    for i in range(len(encodings_list)):
        encodings_list[i]["tsne_encoding"] = [float(tsne_proj[i][0]), float(tsne_proj[i][1])]
    with open(os.path.join(dest_dir, "encodings.json"), "w") as encoding_file:
        json.dump(encodings_list, encoding_file)


def generate_random_walks(model, starting_image, output_root, make_1080=True, add_param_stream=False):
    # try andering with different delta magnitudes
    im = get_image(starting_image, model)
    _, starting_frame = evaluate_model(model, im)
    starting_frame = starting_frame.flatten()
    delta_magnitudes = [10]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 5, 10]
    interpolation_counts = [23]#[0, 1, 5, 11, 23, 47, 119, 239]
    total_frames = 5* 60 * 24
    mode = "colour"
    if model.image_channels == 1:
        mode = "mono"
    scale = "sd"
    if add_param_stream:
        scale = "params"
    elif make_1080:
        scale = "hd"
    for dm in delta_magnitudes:
        for ic in interpolation_counts:
            output_path = os.path.join(output_root, "clamped_random_walk_delta_{}_interp_{}_{}f_{}_{}".format(dm, ic, total_frames, mode, scale))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            frames = wander3(starting_frame, total_frames / (ic + 1), dm)
            if ic > 0:
                frames = interpolate(frames, ic)
            normalized_frames = list(normalize_z_values(frames))
            param_cell_size = 10
            param_frame_count = 36
            for i in range(len(frames)):
                file_path = os.path.join(output_path, "{0:04d}.jpg".format(i))
                generate_image_from_z(model, frames[i], file_path)
                if add_param_stream:
                    start_frame = max(0, i - param_frame_count)
                    frames_to_render = normalized_frames[start_frame:i]
                    # construct param image
                    param_frame = Image.new("RGB", (64* param_cell_size, param_frame_count * param_cell_size))
                    image_drawer = ImageDraw.Draw(param_frame)
                    for pfi in range(len(frames_to_render)):
                        for param_index in range(64):
                            image_drawer.rectangle([
                                param_index*param_cell_size,
                                pfi*param_cell_size,
                                (param_index+1)*param_cell_size,
                                (pfi+1)*param_cell_size], fill=make_colour(frames_to_render[pfi][param_index]))
                    #param_frame = param_frame.convert("RGB")
                    param_frame.save(os.path.join(output_path, "{0:04d}_params.jpg".format(i)))
                    # concat with cloud image
                elif make_1080:
                    resize_image(file_path)
            build_video(output_path)
            if add_param_stream:
                build_video(output_path, "param_video.mp4", "%04d_params.jpg")

def make_colour(value):
    # 0.5 : R327 G201 B14
    # 1.0 : R237 G28  B14
    # 0.0 : R28  G201 B14  
    zero_center = (value * 2.0) - 1.0
    r_mod = 0.0
    g_mod = 0.0
    if zero_center < 0.0:
        r_mod = abs(zero_center) * -209
    if zero_center > 0.0:
        g_mod = abs(zero_center) * -173

    return (237 + int(r_mod), 201 + int(g_mod), 14)

def resize_image(file_path):
    im = Image.open(file_path)
    im = im.resize((1920, 1080), resample=Image.NEAREST, box=(36, 0, 648, 396))
    im.save(file_path)

def build_video(folder_path, output_filename="video.mp4", input_pattern="%04d.jpg"):
    video_name = os.path.join(folder_path, output_filename)
    video_proc = (
            ffmpeg
            .input(folder_path + "/" + input_pattern, pattern_type="sequence", framerate=24)
            .output(video_name)
            .run()
            )

def get_dist(ld, n, alphas):
    result = {}
    loc_param, scale_param = stats.norm.fit(ld[n])
    result["mean"] = stats.norm.mean(loc=loc_param, scale=scale_param)
    result["std"] = stats.norm.std(loc=loc_param, scale=scale_param)
    result["intervals"] = []
    for a in alphas:
        limits = stats.norm.interval(a, loc=loc_param, scale=scale_param)
        result["intervals"].append({"alpha": a, "lower": limits[0], "upper": limits[1]})
    return result


def calculate_encodings(model, source_dir, dest_dir):
    encodings = []
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    normalizer = ImageNormalizer(model.image_width, model.image_height)
    for im, _ in get_images(source_dir, model):
        try:
            normal_images = normalizer.normalize_image(im)
            mode = "colour"
            if model.image_channels == 1:
                mode = "mono"
            for n in normal_images:
                _, z = evaluate_model(model, n[mode])
                encodings.append(z.flatten().tolist())
        except Exception as e:
            print("Failed to transcode {} due to {}".format(filename, str(e)))
    with open(os.path.join(dest_dir, "encodings_{}.json".format(mode)), "w") as encoding_file:
        json.dump(encodings, encoding_file)
    latent_distributions = []
    for i in range(64):
        latent_distributions.append([encodings[k][i] for k in range(len(encodings))])
    parameters = []
    alphas = [0.75, 0.9, 0.95]
    for i in range(len(latent_distributions)):
        parameters.append(get_dist(latent_distributions, i, alphas))
    with open("{}_latent_dimensions.json".format(mode), "w") as param_output:
        json.dump(parameters, param_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-name", action="store", default="cloudvaestorage")
    parser.add_argument("--share-name", action="store", default="training-scratch")
    #parser.add_argument("--directory-name", action="store", default="train-cloud-vae-mono-20190828-031357")
    parser.add_argument("--directory-name", action="store", default="train-cloud-vae-colour-20190828-042617")
    parser.add_argument("--file-name", action="store", default="model-checkpoint-250.pt")
    parser.add_argument("--source-images", action="store")
    args = parser.parse_args()

    account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    output_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "output_transcode")

    model = get_model(args.account_name, account_key, args.share_name, args.directory_name, args.file_name)
    model.eval()
    with torch.no_grad():
        generate_random_walks(model, "C:/data/clouds/17_little__fluffy_clouds.jpeg", "C:/code/cloud/gen_study/output")
        #calculate_encodings(model, "C:/data/clouds/", output_path)
        #transcode_images(model, args.source_images, output_path)