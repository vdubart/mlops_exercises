import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel


def train():
    print("Training day and night")

    # Implement training loop
    model = MyAwesomeModel()
    model.train()
    train_set, _ = mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()

    epochs = 5
    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()

            output = model(images.float())
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_losses += [running_loss / len(train_set)]
        print(
            "Epoch: {}/{}.. ".format(e + 1, epochs),
            "Training Loss: {:.3f}.. ".format(train_losses[-1]),
        )

    print("Saving trained model..")
    torch.save(model.state_dict(), "models/trained_model.pt")
    plt.plot(train_losses)
    plt.savefig("reports/figures/train_loss.png")
    plt.show()


if __name__ == "__main__":
    train()
