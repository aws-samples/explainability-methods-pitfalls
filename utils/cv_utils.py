import glob
import os
import urllib.request
from typing import Union

import requests
import zipfile

import matplotlib.pyplot as plt
import torch

# noinspection PyPep8Naming
import torch.nn.functional as F
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
from tqdm import tqdm
from customResNet import *


_IMAGENET_SAMPLES_FOLDER = "ImagenetSampleImages"


def _imagenet_classes():
    # Define the URL for downloading the ImageNet class names
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    # Download the class names and decode as UTF-8
    with urllib.request.urlopen(url) as f:
        classes = f.read().decode("utf-8")

    # Split the class names into a list of strings
    class_list = classes.split("\n")
    # looks like there are 2 identical classes x 2 (crane and mallot)
    class_list[517] += "-rep"
    class_list[639] += "-rep"
    class_list[12] = "house_finch"
    return class_list


def _imagenet_model(device="cpu"):
    print("Loading ResNet50 model.")
    return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)


def _custom_model(device="cpu"):
    print("Loading custom ResNet50 model.")
    return resnet50custom(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)


def _maybe_download_images(folder_name=_IMAGENET_SAMPLES_FOLDER):
    """
    Downloads some ImageNet sample images if they don't exist.
    :param folder_name: the folder name where the images will be saved.
    :return: None.
    """
    # make folder folder_name if it doesn't exist
    if not os.path.exists(folder_name):
        print("Downloading sample ImageNet images")
        os.mkdir(folder_name)
        repository_url = "https://github.com/EliSchwartz/imagenet-sample-images/archive/refs/heads/master.zip"

        # Download the repository zip file
        response = requests.get(repository_url, stream=True)

        total_size = 109000000  # 109 Mb
        block_size = 1024  # 1KB
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

        # Save the zip file
        zip_file_path = os.path.join(folder_name, "repository.zip")
        with open(zip_file_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        print("Decompressing")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(folder_name)

        # Remove the zip file
        os.remove(zip_file_path)

        # Move the extracted images to the target folder
        extracted_folder = os.path.join(
            folder_name, "imagenet-sample-images-master", ""
        )
        for file_name in os.listdir(extracted_folder):
            source_path = os.path.join(extracted_folder, file_name)
            target_path = os.path.join(folder_name, file_name)
            os.rename(source_path, target_path)

        # Remove the empty extracted folder
        os.rmdir(extracted_folder)

        print("Images downloaded and saved in the 'imagenet-test-images' folder.")


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ImageNetCase:
    def __init__(self):
        # use custom version of ResNet with distinct relu attributes
        self._model = _custom_model()
        self.model.eval()

        # to use torchvision ResNet model, uncomment below
        # self._model = _imagenet_model()

        # standard transformations for the input image
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # inverse transformation
        self._inverse_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

        # transformation with added Gaussian noise
        self._transform_noise = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                AddGaussianNoise(0, 0.15),
            ]
        )

        self._class_names = _imagenet_classes()
        self._current_image = None
        self._current_image_class = None

    @property
    def model(self):
        return self._model

    @property
    def class_names(self):
        if self._class_names:
            return self._class_names

    # load an image with given class name from the folder _IMAGENET_SAMPLES_FOLDER
    def load_image(
        self, class_name_or_index: Union[str, int], folder=_IMAGENET_SAMPLES_FOLDER
    ):
        """
        Loads a sample image of a given class (either given as name or as index)
        """
        _maybe_download_images(folder)
        class_name = (
            self.class_names[class_name_or_index]
            if isinstance(class_name_or_index, int)
            else class_name_or_index
        )
        regex = f"{folder}{os.path.sep}*_{class_name.replace(' ', '_')}.JPEG"
        res = glob.glob(regex)
        try:
            image_path = res[0]
            image = Image.open(image_path).convert("RGB")
            self._current_image = image
            self._current_image_class = class_name
            return image
        except IndexError:
            raise ValueError(f"No image with class name {class_name} found.")

    def predict(
        self,
        image: Union[Image, torch.Tensor] = None,
        return_probabilities=False,
        verbose=False,
    ):
        """
        Predicts the class of an image.

        :param image: the image to predict. If None, the current image is used;
                        if torch.Tensor, assume that it is already transformed and batched.
        :param return_probabilities: whether to return the probabilities instead of the logits.
        :param verbose:

        :return: the predicted class.
        """
        img = self._current_image if image is None else image

        img = (
            self._transform(img).unsqueeze(0)
            if not isinstance(image, torch.Tensor)
            else image
        )

        with torch.no_grad():
            outputs = self.model(
                img.to(self.model.conv1.weight.device)
            )  # these are predicted logits

        if verbose and (image is None or isinstance(image, Image.Image)):
            topk = 5
            probabilities, indices = torch.topk(F.softmax(outputs, dim=1), topk)

            # Print the predicted class label
            print(f"Top-5 predicted classes:")
            for indx, prob in zip(indices[0], probabilities[0]):
                print(f"{self.class_names[indx]}: {prob}")
            if image is None:
                print(f"True class: {self._current_image_class}")

        return F.softmax(outputs, dim=1) if return_probabilities else outputs

    def show_image(self, image: Union[Image, torch.Tensor] = None, figsize=(10, 10)):
        if image is None:
            image = self._current_image
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image[0])  # assume tensors are batched
        plt.figure(figsize=figsize)
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    a = ImageNetCase()
    for i in range(10):
        try:
            a.load_image(i)
            a.show_image()

            a.predict(verbose=True)
            print()
        except Exception as e:
            print(e)
            print()
