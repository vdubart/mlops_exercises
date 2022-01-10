import matplotlib.pyplot as plt
import torch
from torch import nn

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel

import hydra

@hydra.main(config_path="", config_name='config.yaml')
def train(cfg):
    print("Training day and night")

    hparams_model = cfg.hparams_model
    hparams_training = cfg.hparams_training
    #hparams_training = training_cfg.hyperparameters


    torch.manual_seed(hparams_training['seed'])

    # Implement training loop
    model = MyAwesomeModel(hparams_model['input_dim'],
                           hparams_model['hidden_dim1'],
                           hparams_model['hidden_dim2'],
                           hparams_model['hidden_dim3'],
                           hparams_model['output_dim'])
    model.train()
    train_set, _ = mnist(hparams_training['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams_training['lr'])
    criterion = nn.NLLLoss()

    epochs = hparams_training['epochs']
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

    from project_path import PROJECT_PATH

    print("Saving trained model..")
    torch.save(model.state_dict(), str(PROJECT_PATH)+"/models/trained_model.pt")
    plt.plot(train_losses)
    plt.savefig(str(PROJECT_PATH)+"/reports/figures/train_loss.png")
    plt.show()


if __name__ == "__main__":
    train()
