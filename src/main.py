"""
Main State Space Model Training Script
"""
# pylint: disable=import-error disable=wildcard-import disable=eval-used
import argparse
import yaml

from losses import *
from models import *
from logger import *
from datasets import *
from trainers import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State Space Model Training")
    parser.add_argument("config", type=str, help="Path to the config yaml file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    model = eval(config["model_name"])(**config["model_params"])

    dataloaders = eval(config["dataset_name"])(**config["dataset_params"])

    loss = eval(config["loss_name"])(**config["loss_params"])

    logger = eval(config["logger_name"])(config)

    trainer_name = config["trainer_name"]
    trainer_params = config["trainer_params"]
    trainer = eval(trainer_name)(model, dataloaders, loss, logger, trainer_params)
    trainer.run()
