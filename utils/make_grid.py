from PIL import Image
import os
import json
import random


def get_filenames(path):
    for _, _, filenames in os.walk(path):
        return filenames

def main():
    originals_path = "C:/data/clouds"
    transcode_path = "C:/code/cloud_gen_study/output_transcode"
    width_count = 8
    height_count = 6
    buffer_pixels = 216
    width_pixels = 648
    height_pixels = 432
    total_count = width_count * height_count
    total_height = (height_count * height_pixels) + ((height_count + 1) * buffer_pixels)
    total_width = (width_count * width_pixels) + ((width_count + 1) * buffer_pixels)
    transcode_grid = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    original_grid = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    filenames = get_filenames(originals_path)
    random.shuffle(filenames)
    filename_sample = filenames[:total_count]
    for x in range(width_count):
        for y in range(height_count):
            sample_index = x + (y * width_count)
            transcode_image_path = os.path.join(transcode_path, filename_sample[sample_index])
            original_image_path = os.path.join(originals_path, filename_sample[sample_index])
            transcode_image = Image.open(transcode_image_path).resize((width_pixels, height_pixels))
            original_image = Image.open(original_image_path).resize((width_pixels, height_pixels))
            upper_left = (
                (x * width_pixels) + ((x + 1) * buffer_pixels),
                (y * height_pixels) + ((y + 1) * buffer_pixels))
            transcode_grid.paste(transcode_image, upper_left)
            original_grid.paste(original_image, upper_left)
    transcode_grid.save("transcode_grid.png")
    original_grid.save("original_grid.png")


if __name__ == "__main__":
    main()