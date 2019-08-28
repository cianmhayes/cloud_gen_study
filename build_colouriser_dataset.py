from azure.storage.blob import BlockBlobService
import os
from datetime import datetime
import sys
from PIL import Image
from utils.image_normalizer import ImageNormalizer
from uuid import uuid4

def prepare_datset(working_folder, source_container, destination_container):
    blob_service = BlockBlobService(account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"], account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"])
    print("creating container {}".format(destination_container))
    blob_service.create_container(destination_container)
    temp_image_folder = os.path.join(working_folder, "temp")
    colour_image_folder = os.path.join(working_folder, "colour")
    monochrome_image_folder = os.path.join(working_folder, "mono")
    normalizer = ImageNormalizer(648, 432)
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)
    if not os.path.exists(colour_image_folder):
        os.makedirs(colour_image_folder)
    if not os.path.exists(monochrome_image_folder):
        os.makedirs(monochrome_image_folder)
    
    for blob_name in blob_service.list_blob_names(container_name=source_container):
        temp_image_local_path = os.path.join(temp_image_folder, blob_name)
        blob_service.get_blob_to_path(source_container, blob_name, temp_image_local_path)
        name, _ = os.path.splitext(blob_name)
        im = Image.open(temp_image_local_path)
        for (colour, mono) in normalizer.normalize_image(im):
            normalized_file_name = "{}.{}.png".format(name, uuid4())
            colour_image_local_path = os.path.join(colour_image_folder, normalized_file_name)
            monochrome_image_local_path = os.path.join(monochrome_image_folder, normalized_file_name)
            colour.save(colour_image_local_path)
            mono.save(monochrome_image_local_path)
            blob_service.create_blob_from_path(destination_container, "colour/"+normalized_file_name, colour_image_local_path)
            blob_service.create_blob_from_path(destination_container, "mono/"+normalized_file_name, monochrome_image_local_path)

if __name__ == "__main__":
    target_container = "colouriser-training-data-w648-h432-{:%Y%m%d%H%M}".format(datetime.utcnow())
    prepare_datset("./temp/", "original-images", target_container)
    
