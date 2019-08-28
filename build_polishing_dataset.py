
from models.cloud_conv_vae import load_model, CloudConvVae
from azure.storage.blob import BlockBlobService
import os
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from datetime import datetime
import sys
from PIL import Image

def prepare_datset(working_folder, source_container, destination_container, model_path):
    model = load_model(model_path)
    model.eval()
    blob_service = BlockBlobService(account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"], account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"])
    blob_service.create_container(destination_container)
    original_image_folder = os.path.join(working_folder, "original")
    reconstructed_image_folder = os.path.join(working_folder, "reconstructed")
    if not os.path.exists(original_image_folder):
        os.makedirs(original_image_folder)
    if not os.path.exists(reconstructed_image_folder):
        os.makedirs(reconstructed_image_folder)
    
    for blob_name in blob_service.list_blob_names(container_name=source_container):
        original_image_local_path = os.path.join(original_image_folder, blob_name)
        reconstructed_image_local_path = os.path.join(reconstructed_image_folder, blob_name)
        blob_service.get_blob_to_path(source_container, blob_name, original_image_local_path)

        orginal_image = Image.open(original_image_local_path)
        original_image_tensor = ToTensor()(orginal_image).view(-1, 1, 432, 648)
        reconstructed_tensor = model(original_image_tensor)[0]
        save_image(reconstructed_tensor, reconstructed_image_local_path)

        blob_service.create_blob_from_path(destination_container, "original/"+blob_name, original_image_local_path)
        blob_service.create_blob_from_path(destination_container, "reconstructed/"+blob_name, reconstructed_image_local_path)



if __name__ == "__main__":
    target_container = "polisher-training-data-w648-h432-{:%Y%m%d%H%M}".format(datetime.utcnow())
    prepare_datset("./temp/", "training-data-w648-h432-201907131642", target_container, "model-checkpoint-200.pt")
    
