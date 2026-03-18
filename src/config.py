from dataclasses import dataclass
import yaml


@dataclass
class Config:
    experiment_name: str
    run_name: str
    seed: int
    image_size: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    freeze_backbone_epochs: int
    early_stopping_patience: int
    mlflow_tracking_uri: str
    output_dir: str


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)