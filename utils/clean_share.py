import argparse
import os
import uuid
from azure.storage.file.fileservice import FileService

if __name__ == "__main__":
    fs = FileService(
        account_name=os.environ["AZURE_STORAGE_ACCOUNT_NAME"],
        account_key=os.environ["AZURE_STORAGE_ACCOUNT_KEY"])
    
    for f in fs.list_directories_and_files("training-scratch","train-cloud-vae-20190705-024325"):
        fs.delete_file("training-scratch","train-cloud-vae-20190705-024325",f.name)

    
    for f in fs.list_directories_and_files("training-scratch","train-cloud-vae-20190705-205925"):
        if f.name.startswith("model-checkpoint-"):
            try:
                epoch = int(f.name.replace("model-checkpoint-", "").replace(".pt", ""))
                if epoch not in [51, 76, 101, 501, 999]:
                    print("delete checkpoint for epoch {}".format(epoch))
                    fs.delete_file("training-scratch","train-cloud-vae-20190705-205925",f.name)
            except Exception as e:
                print(e)