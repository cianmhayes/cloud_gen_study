import os
import json
import numpy as np
from datetime import datetime
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from azure.storage.blob import BlockBlobService

from datasets.image_dataset import ImageDataset
from models.cloud_conv_vae import CloudConvVae, save_model
from utils.instrumentation import ScriptInstrumentor

np.random.seed(20190629)

class CloudVaeTrainer(object):
    def __init__(
            self,
            dataset_path,
            image_set_name,
            working_directory,
            image_limit=None,
            starting_learning_rate= 0.001,
            gradient_clip = None,
            batch_size=16,
            validation_split=0.2,
            hidden=640,
            latent=64,
            debug=False,
            force_parallel=False,
            force_cpu=False):
        self.run_name = "train-cloud-vae-{0}-{1:%Y%m%d}-{1:%H%M%S}".format(image_set_name, datetime.utcnow())
        self.image_set_name = image_set_name
        self.hidden = hidden
        self.latent = latent
        self.trace_memory = False
        self.debug = debug

        self.gradient_clip = gradient_clip

        self.location = os.path.join(working_directory, self.run_name)
        self.tb_writer = SummaryWriter(self.location)

        self.instrumentor = ScriptInstrumentor(
            {
                "model_class_name": "CloudConvVae",
                "run_name": self.run_name,
                "image_set_name": image_set_name,
                "batch_size": batch_size
                })

        self.instrumentor.start_dataset_load()
        self.dataset = ImageDataset(dataset_path, [image_set_name], image_limit)
        self.instrumentor.end_dataset_load()

        self.width = self.dataset.get_image_width(image_set_name)
        self.height = self.dataset.get_image_height(image_set_name)
        self.channels = self.dataset.get_image_channels(image_set_name)
        
        device_name = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        self.instrumentor.set_context("device", device_name)
        self.device = torch.device(device_name)
        self.model = CloudConvVae(self.width, self.height, hidden, latent, self.channels)
        self.pin_memory = False
        if torch.cuda.device_count() > 1 and not force_cpu:
            print("Found {} gpus".format(torch.cuda.device_count()))
            self.instrumentor.set_context("gpu_count", str(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
            self.pin_memory = True
        elif force_parallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=starting_learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 40, gamma=0.1)
        self.initial_sample = None
        self.storage_account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
        self.storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
        self.blob_storage_client = BlockBlobService(self.storage_account_name, self.storage_account_key)
        
        self.instrumentor.start_dataset_split()
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices = indices[split:]
        validation_indices = indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(validation_indices)
        self.training_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=self.pin_memory)
        self.validation_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            pin_memory=self.pin_memory)
        self.instrumentor.end_dataset_split()

    def run(self, epochs):
        self.blob_storage_client.create_container(self.run_name)
        for epoch in range(1, epochs+1):
            self.instrumentor.start_epoch(epoch)
            print("==================================================")
            print("Epoch {}".format(epoch))
            try:
                self._train(epoch)
                self._test(epoch)
                if self._should_save_state(epoch):
                    self._save_state(epoch)
                self.lr_scheduler.step()
            except Exception as e:
                self.instrumentor.log_exception({})
                print(e)
            self.instrumentor.end_epoch()

    def loss_function(self, decoded_values, values, mu, log_variance):
        cross_entropy = F.binary_cross_entropy(decoded_values, values, reduction="sum")
        kl_divergence = -0.5 * torch.sum(1+ log_variance - mu.pow(2) - log_variance.exp())
        return cross_entropy + kl_divergence, cross_entropy, kl_divergence

    def _should_save_state(self, epoch):
        return epoch < 50 or (epoch < 100 and epoch % 10 == 0) or epoch % 50 == 0

    def _save_state(self, epoch):
        local_output_path = os.path.join(self.location, "model-checkpoint-{}.pt".format(epoch))
        save_model(local_output_path, self.model, epoch=epoch, optimizer=self.optimizer)

    def _trace_memory(self, reset_maxes = True):
        if torch.cuda.is_available() and self.trace_memory:
            self.instrumentor.trace_memory(
                {
                    "max_memory_cached": torch.cuda.max_memory_cached(device=self.device),
                    "max_memory_allocated": torch.cuda.max_memory_allocated(device=self.device),
                    "memory_allocated": torch.cuda.memory_allocated(device=self.device),
                    "memory_cached": torch.cuda.memory_cached(device=self.device)
                    })
            if reset_maxes:
                torch.cuda.reset_max_memory_allocated(device=self.device)
                torch.cuda.reset_max_memory_cached(device=self.device)

    def _train(self, epoch):
        self.instrumentor.start_phase("train")
        self.model.train()
        train_loss = 0
        for index, (image, name) in enumerate(self.training_data_loader):
            decoded_image = None
            try:
                image = image.to(self.device, non_blocking=self.pin_memory)
                self.optimizer.zero_grad()
                decoded_image, mu, log_variance, z = self.model(image)
                loss = self.loss_function(decoded_image, image, mu, log_variance)[0]
                loss.backward()
                if self.gradient_clip is not None:
                    # https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/16
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                train_loss += float(loss)
                self._trace_memory()
                del image, decoded_image, mu, log_variance, z
            except Exception as e:
                self._log_tensor_error(index, "train", image, decoded_image, mu, log_variance, z, name)
                print("Error: {}".format(str(e)))
        train_loss /= len(self.training_data_loader.dataset)
        self.tb_writer.add_scalar("train_loss", train_loss, epoch)
        print('Train set loss: {:.4f}'.format(train_loss))
        self.instrumentor.end_phase("train", train_loss)

    def _test(self, epoch):
        self.instrumentor.start_phase("test")
        self.model.eval()
        test_loss = 0
        test_kl_dviergence_loss = 0
        test_cross_entropy_loss = 0
        with torch.no_grad():
            for index, (image, name) in enumerate(self.validation_data_loader):
                decoded_image = None
                try:
                    image = image.to(self.device, non_blocking=self.pin_memory)
                    decoded_image, mu, log_variance, z = self.model(image)
                    loss, cross_entropy, kl_divergence = self.loss_function(decoded_image, image, mu, log_variance)
                    test_loss += float(loss)
                    test_cross_entropy_loss += float(cross_entropy)
                    test_kl_dviergence_loss += float(kl_divergence)
                    for i in range(len(decoded_image)):
                        try:
                            file_name = "decoded_test_images_e_{}_{}.png".format(epoch, name[i])
                            blob_name = "decoded_test_images/e_{}/{}.png".format(epoch, name[i])
                            save_image(decoded_image[i], file_name)
                            self.blob_storage_client.create_blob_from_path(self.run_name, blob_name, file_name)
                            os.remove(file_name)
                        except Exception as e:
                            print(str(e))
                    del image, decoded_image, mu, log_variance, z
                except Exception as e2:
                    self._log_tensor_error(index, "test", image, decoded_image, mu, log_variance, z, name)
                    print("Error: {}".format(str(e2)))
        test_loss /= len(self.validation_data_loader.dataset)
        test_cross_entropy_loss /= len(self.validation_data_loader.dataset)
        test_kl_dviergence_loss /= len(self.validation_data_loader.dataset)
        self.tb_writer.add_scalar("test_loss", test_loss, epoch)
        self.tb_writer.add_scalar("test_cross_entropy_loss", test_cross_entropy_loss, epoch)
        self.tb_writer.add_scalar("test_kl_dviergence_loss", test_kl_dviergence_loss, epoch)
        print('Test set loss: {:.4f}'.format(test_loss))
        self.instrumentor.end_phase("test", test_loss)

    def _log_tensor_error(self, batch, phase, images, decoded_images, mu, log_variance, z, names):
        bad_images = []
        for i in range(len(images)):
            image_summary = {}
            image_summary["name"] = names[i]
            min_value = float(torch.min(images[i]))
            max_value = float(torch.max(images[i]))
            image_summary["image_min"] = min_value
            image_summary["image_max"] = max_value
            if decoded_images is not None:
                min_value = float(torch.min(decoded_images[i]))
                max_value = float(torch.max(decoded_images[i]))
                image_summary["decoded_image_min"] = min_value
                image_summary["decoded_image_max"] = max_value
            if mu is not None:
                min_value = float(torch.min(mu[i]))
                max_value = float(torch.max(mu[i]))
                image_summary["mu_min"] = min_value
                image_summary["mu_max"] = max_value
            if log_variance is not None:
                min_value = float(torch.min(log_variance[i]))
                max_value = float(torch.max(log_variance[i]))
                image_summary["log_variance_min"] = min_value
                image_summary["log_variance_max"] = max_value
            if z is not None:
                min_value = float(torch.min(z[i]))
                max_value = float(torch.max(z[i]))
                image_summary["z_min"] = min_value
                image_summary["z_max"] = max_value
            bad_images.append(image_summary)
        self.instrumentor.log_exception(
            {
                "phase": phase,
                "images": json.dumps(bad_images),
                "batch": batch
                })

    def _sample_image(self, epoch, quantity=16):
        with torch.no_grad():
            sample = torch.randn(quantity, self.latent).to(self.device)
            if self.initial_sample is None:
                self.initial_sample = sample

            sample = self.model.decode(sample).cpu()
            sample_set = sample.view(quantity, self.channels, self.height, self.width)
            for i in range(quantity):
                file_name = "image_samples_e_{}_i_{}.png".format(epoch, i)
                blob_name = "image_samples/e_{}/i_{}.png".format(epoch, i)
                save_image(sample_set[i], file_name)
                self.blob_storage_client.create_blob_from_path(self.run_name, blob_name, file_name)
                os.remove(file_name)

            updated_initial_sample = self.model.decode(self.initial_sample).cpu()
            updated_initial_sample_set = updated_initial_sample.view(quantity, self.channels, self.height, self.width)
            for i in range(quantity):
                file_name = "initial_image_samples_e_{}_i_{}.png".format(epoch, i)
                blob_name = "initial_image_samples/e_{}/i_{}.png".format(epoch, i)
                save_image(updated_initial_sample_set[i], file_name)
                self.blob_storage_client.create_blob_from_path(self.run_name, blob_name, file_name)
                os.remove(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", action="store", type=int, default=1000)
    parser.add_argument("--output-path", action="store", default="./output/")
    parser.add_argument("--data-path", action="store")
    parser.add_argument("--image-set", action="store")
    parser.add_argument("--limit", action="store", type=int)
    parser.add_argument("--learning-rate", action="store", default=0.001, type=float)
    parser.add_argument("--batch-size", action="store", default=16, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force-parallel", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--gradient-clip", action="store", type=float)
    args = parser.parse_args()

    trainer = CloudVaeTrainer(
        args.data_path, args.image_set, args.output_path,
        image_limit=args.limit,
        starting_learning_rate=args.learning_rate,
        gradient_clip=args.gradient_clip,
        batch_size=args.batch_size,
        debug=args.debug,
        force_parallel=args.force_parallel,
        force_cpu=args.force_cpu)
    print("Start training")
    trainer.run(args.epoch)
    print("Done")