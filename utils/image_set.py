import argparse
import os
import uuid
from azure.storage.blob import BlockBlobService
from azure.cosmosdb.table import TableService
from datetime import datetime
from PIL import Image
from image_normalizer import ImageNormalizer
import lmdb
import json
from torchvision.transforms import ToTensor
import numpy as np

one_gb = 1024 * 1024 * 1024

class ImageSet(object):
    def __init__(
            self,
            storge_account_name,
            original_image_metadata_table="OriginalImageMetadata",
            original_image_blob_container="original-images"):
        self.blob_service = None
        self.table_Service = None
        self.storage_account_name = storge_account_name
        self.original_image_metadata_table = original_image_metadata_table
        self.original_image_blob_container = original_image_blob_container

    def load_original_images(self, image_folder, label = None):
        blob_service = self._get_or_create_blob_service()
        table_Service = self._get_or_create_table_service()

        for dirpath, dirname, file_names in os.walk(image_folder):
            for file_name in file_names:
                path = os.path.join(dirpath, file_name)
                image_id = str(uuid.uuid4())
                blob_name = "{}.png".format(image_id)
                im = Image.open(path)
                im.save(blob_name)
                entity = {"PartitionKey": image_id, "RowKey": image_id, "SetLabel": label, "OriginalName": file_name, "Width": im.size[0], "Height": im.size[1], "Bands": "".join(im.getbands())}
                blob_service.create_blob_from_path(self.original_image_blob_container, blob_name, blob_name)
                table_Service.insert_or_replace_entity(self.original_image_metadata_table, entity)
                print(entity)
                os.remove(blob_name)

    def create_normalized_dataset(self, target_width, target_height, root_path, max_db_gb=2, image_limit=None):
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        normalizer = ImageNormalizer(target_width, target_height)
        mono_table_name = "mono"
        colour_table_name = "colour"
        index_table_name = "index"
        meatadata = {
            mono_table_name: {
                "width": target_width,
                "height": target_height,
                "channels": 1,
                "dtype": "float32"
            },
            colour_table_name: {
                "width": target_width,
                "height": target_height,
                "channels": 3,
                "dtype": "float32"
            },
            index_table_name: {
                "encoding": "ascii",
                "dtype": "string"
            }
        }
        with open(os.path.join(root_path, "metadata.json"), "w") as metadata_file:
            json.dump(meatadata, metadata_file)
        env = lmdb.open(root_path, map_size=(max_db_gb * one_gb), create=True, max_dbs=4)
        mono_db = env.open_db(mono_table_name.encode(encoding="ascii"))
        colour_db = env.open_db(colour_table_name.encode(encoding="ascii"))
        index_db = env.open_db(index_table_name.encode(encoding="ascii"))
        i = 0
        blob_service = self._get_or_create_blob_service()
        blobs_to_be_normalized = blob_service.list_blob_names(self.original_image_blob_container, num_results=image_limit)
        source_count = len(list(blobs_to_be_normalized))
        source_index = 0
        for original_blob_name in blobs_to_be_normalized :
            source_index += 1
            print("Normalizing {} of {} : {}".format(source_index, source_count, original_blob_name))
            self.blob_service.get_blob_to_path(self.original_image_blob_container, original_blob_name, original_blob_name)
            im = Image.open(original_blob_name)
            image_id = original_blob_name.split(".")[0]
            normalized = normalizer.normalize_image(im)
            for normal_image in normalized:
                i_key = i.to_bytes(4,byteorder="big")
                i += 1
                new_image_id = "{0}.{1}.png".format(image_id, str(uuid.uuid4()))
                mono_image_array = ToTensor()(normal_image["mono"]).flatten().numpy()
                colour_image_array = ToTensor()(normal_image["colour"]).flatten().numpy()
                # insert in db
                with env.begin(db=index_db, write=True, buffers=True) as txn:
                    txn.put(i_key, new_image_id.encode(encoding="ascii"))
                with env.begin(db=mono_db, write=True, buffers=True) as txn:
                    txn.put(i_key, mono_image_array.data)
                with env.begin(db=colour_db, write=True, buffers=True) as txn:
                    txn.put(i_key, colour_image_array.data)
            print("{} normalized images for {}".format(len(normalized), original_blob_name))
            os.remove(original_blob_name)
        env.close()
        # open and close it again as readonly to compress the file
        env2 = lmdb.open(root_path, max_dbs=1)
        env2.close()

    def _get_or_create_blob_service(self):
        if self.blob_service is None:
            self.blob_service = BlockBlobService(account_name=self.storage_account_name, account_key=self._get_account_key())
        return self.blob_service
    
    def _get_or_create_table_service(self):
        if self.table_Service is None:
            self.table_Service = TableService(account_name=self.storage_account_name, account_key=self._get_account_key())
        return self.table_Service

    def _get_account_key(self):
        return os.environ["AZURE_STORAGE_ACCOUNT_KEY"]

def load_images(args):
    image_set = ImageSet("cloudvaestorage")
    image_set.load_original_images(args.path, args.label)

def normalize_images(args):
    image_set = ImageSet("cloudvaestorage")
    image_set.create_normalized_dataset(args.width, args.height, args.output, args.db_size, args.limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    load_command = subparsers.add_parser("load")
    load_command.add_argument(
        "-p",
        "--path",
        action="store",
        required=True)
    load_command.add_argument(
        "-l",
        "--label",
        action="store")
    load_command.set_defaults(func=load_images)

    normalize_command = subparsers.add_parser("normalize")
    normalize_command.add_argument(
        "--width",
        type=int,
        default=720,
        action="store")
    normalize_command.add_argument(
        "--height",
        type=int,
        default=480,
        action="store")
    normalize_command.add_argument(
        "-o",
        "--output",
        action="store")
    normalize_command.add_argument(
        "-d",
        "--db-size",
        type=int,
        default=2,
        action="store")
    normalize_command.add_argument(
        "-l",
        "--limit",
        type=int,
        action="store")
    normalize_command.set_defaults(func=normalize_images)

    args = parser.parse_args()
    args.func(args)