import json
import os
import math
from PIL import Image
import numpy as np

def get_encoding_data():
    with open("C:/code/cloud_gen_study/output_transcode/encodings.json", "r") as encoding_file:
        return json.load(encoding_file)

def sign(f):
    if f < 0.0:
        return -1.0
    return 1.0

def layout_images_on_grid():
    encodings = get_encoding_data()
    encodings.sort(key= lambda x : x["tsne_encoding"][0])
    print(encodings[0]["tsne_encoding"][0])
    print(encodings[1]["tsne_encoding"][0])
    print(encodings[2]["tsne_encoding"][0])

def main():
    encodings = get_encoding_data()
    deltas_0 = []
    deltas_1 = []
    distances = []
    for enc1 in encodings:
        for enc2 in encodings:
            if enc1["file_name"] == enc2["file_name"]:
                continue
            delta_0 = (enc1["tsne_encoding"][0] - enc2["tsne_encoding"][0])
            delta_1 = (enc1["tsne_encoding"][1] - enc2["tsne_encoding"][1])
            deltas_0.append(abs(delta_0))
            deltas_1.append(abs(delta_1))
            distance = math.sqrt((delta_0 ** 2) + (delta_1 ** 2))
            distances.append(distance)
    print("min delta 0: {}".format(min(deltas_0)))
    print("min delta 1: {}".format(min(deltas_1)))
    print("min distances: {}".format(min(distances)))
    print("max delta 0: {}".format(max(deltas_0)))
    print("max delta 1: {}".format(max(deltas_1)))
    print("max distances: {}".format(max(distances)))

    log_deltas_0 = [math.log2(d*100000) for d in deltas_0]
    log_deltas_1 = [math.log2(d*100000) for d in deltas_1]
    log_distances = [math.log2(d*100000) for d in distances]
    print("min log delta 0: {}".format(min(log_deltas_0)))
    print("min log delta 1: {}".format(min(log_deltas_1)))
    print("min log distances: {}".format(min(log_distances)))
    print("max log delta 0: {}".format(max(log_deltas_0)))
    print("max log delta 1: {}".format(max(log_deltas_1)))
    print("max log distances: {}".format(max(log_distances)))



    

if __name__ == "__main__":
    #main()
    layout_images_on_grid()