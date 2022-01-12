import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
import os

from src.data.make_dataset import mnist
from src.models.modelLightning import MyAwesomeModel

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


    train_set, _ = mnist(hparams_training['batch_size'])

    trainer = Trainer(default_root_dir=os.getcwd(),
                      max_epochs=10)
    trainer.fit(model, train_set)


    #from project_path import PROJECT_PATH
    #print("Saving trained model..")
    #torch.save(model.state_dict(), str(PROJECT_PATH)+"/models/trained_model.pt")
    #plt.plot(train_losses)
    #plt.savefig(str(PROJECT_PATH)+"/reports/figures/train_loss.png")
    #plt.show()


if __name__ == "__main__":
    train()
