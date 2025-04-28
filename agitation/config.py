"""Module defining common configuration as constant variables."""

import os

from platformdirs import user_data_dir

DATA_DIR = user_data_dir("agitation")
MODEL_PATH = os.path.join(DATA_DIR, "best_model.pt")
ZENODO_URL = "https://zenodo.org/records/15288225/files/best_torchscript.pt?download=1"
ZENODO_MD5 = "1836dcfc6e264a8630e4312565d4e51f"


## MOTION MM PARAMETERS
MOTION_N_BINS = 50
MOTION_BIN_RANGE = (-0.5, 4.5)
VOLUME_SHAPE = (160, 192, 160)
