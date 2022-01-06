import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from src.data.make_dataset import mnist
from src.models.model import MyAwesomeModel


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("visualization_filepath", type=click.Path())
def visualize(model_filepath, visualization_filepath):
    """
    Visualize the features of the third layer of a pre-trained model on training set
    on a 2D plane using t-SNE dimension reduction.

            Parameters:
                    model_filepath (str): Path to the pre-trained model parameters
                    visualization_filepath (str): Path and name of the resulting plot
    """
    print("Visualizing")

    # Construct model from pre-trained parameters in saved file
    model = MyAwesomeModel()
    state_dict = torch.load(model_filepath)
    model.load_state_dict(state_dict)

    train_set, _ = mnist()

    X = []
    colors = []
    with torch.no_grad():
        for images, labels in train_set:
            output, features = model.forward_extract(images.float())

            # Keep track of the feature for each individual image in the batch
            for feature in features:
                X.append(feature.numpy())
            # Keep track of the class
            for label in labels:
                colors.append(label)

    X = np.array(X)
    print(X.shape)

    # Apply t-SNE in 2D to reduce dimensionality of features (64D)
    X_embedded = TSNE(
        n_components=2, perplexity=100, learning_rate="auto", init="random"
    ).fit_transform(X)
    print(X_embedded.shape)

    # Create figure
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors, cmap="gist_rainbow")
    plt.title(
        "Layer 3's features in a 2D space using t-SNE \n (dots colored by class label)"
    )
    plt.savefig(visualization_filepath)
    return


if __name__ == "__main__":
    visualize()
