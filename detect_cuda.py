import torch

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Found {} CUDA devices".format(torch.cuda.device_count()))
    else:
        print("CUDA not available")