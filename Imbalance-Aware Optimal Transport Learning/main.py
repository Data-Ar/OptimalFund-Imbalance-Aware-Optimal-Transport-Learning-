"""Main entrypoint for DR severity benchmark."""

import torch

from util.config import cfg
from dataset.loaders import build_data_objects
from train.sweep import run_all_ot_modes


if __name__ == "__main__":
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    data_objects = build_data_objects(cfg)

    print(
        "Hospital images:",
        len(data_objects['ds_h_train']),
        len(data_objects['ds_h_val']),
        len(data_objects['ds_h_test']),
    )
    print(
        "Phone    images:",
        len(data_objects['ds_p_train']),
        len(data_objects['ds_p_val']),
        len(data_objects['ds_p_test']),
    )

    run_all_ot_modes(cfg, data_objects)
