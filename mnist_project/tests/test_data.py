import torch
from src.data.make_dataset import mnist
from project_path import PROJECT_PATH

def test_data():

    train_images = torch.load(str(PROJECT_PATH)+"/data/processed/train_0.pt")
    train_labels = torch.load(str(PROJECT_PATH)+"/data/processed/train_0_labels.pt")
    assert len(train_images) == len(train_labels)
    N_train = len(train_images)

    test_images = torch.load(str(PROJECT_PATH)+"/data/processed/test.pt")
    test_labels = torch.load(str(PROJECT_PATH)+"/data/processed/test_labels.pt")
    assert len(test_images) == len(test_labels)
    N_test = len(test_images)

    trainloader, testloader = mnist()

    # A) assert len(dataset) == N_train for training and N_test for test
    n_train = 0
    for images, labels in trainloader:
        for image in images:
            n_train += 1
    n_test = 0
    for images, labels in testloader:
        for image in images:
            n_test += 1

    assert n_train == N_train
    assert n_test == N_test

    # B) assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    for images, labels in trainloader:
        for image in images:
            assert image.shape == (28,28)
    

    # C) assert that all labels are represented
    unique_labels = torch.unique(train_labels)
    # ...?
