# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    example data ready to be analyzed/predicted saved in (data/)
    """
    logger = logging.getLogger(__name__)
    logger.info("making example data from raw data")

    data = np.load(input_filepath)
    x = data["images"][0:10]
    np.save(output_filepath, x)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
