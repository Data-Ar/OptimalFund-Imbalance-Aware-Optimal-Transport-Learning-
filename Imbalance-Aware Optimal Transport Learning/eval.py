"""Evaluation entrypoint for DR severity benchmark."""

from util.config import cfg
from dataset.loaders import build_data_objects
from util.metrics import evaluate
from model.dr_model import DRModel


def main():
    data_objects = build_data_objects(cfg)
    model = DRModel(cfg.backbones[0], cfg.num_classes, use_feature_norm=cfg.use_feature_norm).to(cfg.device)
    macro_auc, *_ = evaluate(model, data_objects["ld_p_test"])
    print(f"Untrained model macro AUC on clean phone test: {macro_auc:.4f}")


if __name__ == "__main__":
    main()
