# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from torchvision import transforms

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    data = np.load(input_filepath)
    images_tensors = torch.zeros(data["images"].shape)
    for i, image in enumerate(data["images"]):
        images_tensors[i] = transform(image)
    torch.save(images_tensors, output_filepath + ".pt")

    labels = torch.from_numpy(data["labels"])
    torch.save(labels, output_filepath + "_labels.pt")


class CustomImageDataset(Dataset):
    def __init__(self, filepath):
        self.images = torch.load(filepath + ".pt")
        self.labels = torch.load(filepath + "_labels.pt")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def mnist(batch_size=64):
    # exchange with the corrupted mnist dataset
 
    from project_path import PROJECT_PATH

    trainset = CustomImageDataset(filepath=str(PROJECT_PATH)+"/data/processed/train_0")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = CustomImageDataset(filepath=str(PROJECT_PATH)+"/data/processed/test")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
