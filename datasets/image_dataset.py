import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
import json
import lmdb
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_path, target_sets, image_limit=None):
        self.target_sets = target_sets
        self.set_metadata = {}
        self.tensor_shapes = {}

        self.db_env = lmdb.open(root_path, max_dbs=len(target_sets)+1, readonly=True, lock=False)
        self.db_env.reader_check()
        self.sub_dbs = {}
        self.image_count = None
        self.image_limit = image_limit

        with open(os.path.join(root_path, "metadata.json"), "r") as metadata_file:
            metadata = json.load(metadata_file)
            self.index_db = None
            if "index" in metadata:
                self.index_db = self.db_env.open_db("index".encode(encoding="ascii"))
                self.image_count = self._get_db_count(self.index_db)
            for t in target_sets:
                if t not in metadata:
                    raise Exception("Dataset does not contain image set '{}'".format(t))
                if "width" not in metadata[t]:
                    raise Exception("Image set '{}' does not have a width property".format(t))
                if "height" not in metadata[t]:
                    raise Exception("Image set '{}' does not have a height property".format(t))
                if "channels" not in metadata[t]:
                    raise Exception("Image set '{}' does not have a channels property".format(t))
                self.set_metadata[t] = metadata[t]
                self.tensor_shapes[t] = (metadata[t]["channels"], metadata[t]["height"], metadata[t]["width"])
                self.sub_dbs[t] = self.db_env.open_db(t.encode(encoding="ascii"))
                current_set_length = self._get_db_count(self.sub_dbs[t])
                if self.image_count is None:
                    self.image_count = current_set_length
                if not (self.image_count == current_set_length):
                    raise Exception("Expected {} entries in {} but got {}".format(self.image_count, t, current_set_length))

    def _get_db_count(self, db):
        with self.db_env.begin() as txn:
            c = txn.cursor(db)
            return txn.stat(db)['entries']

    def get_image_width(self, set_name):
        return self.set_metadata[set_name]["width"]

    def get_image_height(self, set_name):
        return self.set_metadata[set_name]["height"]

    def get_image_channels(self, set_name):
        return self.set_metadata[set_name]["channels"]

    def __len__(self):
        return self.image_limit or self.image_count

    def __getitem__(self, idx):
        result = []
        idx_key = int(idx).to_bytes(4,byteorder="big")
        for target in self.sub_dbs.keys():
            with self.db_env.begin() as txn:
                c = txn.cursor(self.sub_dbs[target])
                value = c.get(idx_key)
                if value is None:
                    raise Exception("Could not find element with key '{}' in {}".format(idx, target))
                np_value = np.frombuffer(value, dtype=np.dtype(np.float32))
                tensor_value = torch.from_numpy(np_value)
                tensor_value = torch.reshape(tensor_value, self.tensor_shapes[target])
                result.append(tensor_value)
        with self.db_env.begin() as txn:
            c = txn.cursor(self.index_db)
            value = c.get(idx_key).decode(encoding="ascii")
            result.append(value)
        return tuple(result)
