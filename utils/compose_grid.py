import math
import os
from PIL import Image
import re
from azure.storage.blob import BlockBlobService
import uuid

def construct_grid(storage_account_name, container_name, image_set):
    blob_client = BlockBlobService(account_name=storage_account_name, account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"])
    blobs = list(blob_client.list_blob_names(container_name, image_set))

    images = {}

    min_epoch = None
    for blob_name in blobs:
        blob_name_parts = blob_name.split("/")
        file_name = blob_name_parts[2]
        epoch = int(blob_name_parts[1].strip("e_"))
        if file_name not in images.keys():
            images[file_name] = {}
        images[file_name][epoch] = blob_name

    partial_epoch_images = len(blobs) % len(images.keys())
    target_epochs = int(math.floor((len(blobs) - partial_epoch_images) / len(images.keys())))
    left = 0
    top = 0
    width = 383
    height = 256
    image_group_count = len(images.keys())
    composite_image = Image.new("L", (image_group_count * width, target_epochs * height))
    for image_name in images.keys():
        top = 0
        print("Starting {}".format(image_name))
        for epoch in range(1, target_epochs+1):
            try:
                tmp_file = "{}.png".format(uuid.uuid4())
                blob_client.get_blob_to_path(container_name, images[image_name][epoch], tmp_file)
                #image_bytes = blob_client.get_blob_to_bytes(container_name, images[image_name][epoch])
                #im = Image.frombytes("L", (width, height), image_bytes.content)
                im = Image.open(tmp_file)
                composite_image.paste(im, box=(left,top))
                os.remove(tmp_file)
            except Exception as e:
                print("Error incorporating {} due to : {}".format(images[image_name][epoch], str(e)))
            top += height
        left += width
    return composite_image

if __name__ == "__main__":
    decode_grid = construct_grid("cloudvaestorage", "train-cloud-vae-20190705-205925", "decoded_test_images")
    decode_grid.save("train-cloud-vae-20190705-205925.decoded_test_images.png")