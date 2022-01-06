import click
import numpy as np
import torch

from src.models.model import MyAwesomeModel


@click.command()
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("example_images_file", type=click.Path(exists=True))
def predict(model_filepath, example_images_file):
    """
    Predict the classes for some unlabelled data using a pre-trained model.

            Parameters:
                    model_filepath (str): Path to the pre-trained model parameters
                    example_images_file (str): Path to the examples images data

            Returns:
                    predicted_classes (list): Predicted classes of the images
    """
    print("Predicting")

    # Construct model from pre-trained parameters in saved file
    model = MyAwesomeModel()
    state_dict = torch.load(model_filepath)
    model.load_state_dict(state_dict)

    example_images = np.load(example_images_file)

    predicted_classes = []
    with torch.no_grad():
        # set model to evaluation mode
        model.eval()

        # Prediction pass
        for i, image in enumerate(example_images):
            # Get the class probabilities
            log_ps = model(torch.from_numpy(image).float().unsqueeze(0))
            ps = torch.exp(log_ps)

            # Get top probability and class
            top_p, top_class = ps.topk(1, dim=1)
            predicted_classes.append(top_class.item())
            print(
                "Image {} : predicted class {} (with probability {:.3f}%)".format(
                    i, top_class.item(), top_p.item()
                )
            )

    return predicted_classes


if __name__ == "__main__":
    predict()
