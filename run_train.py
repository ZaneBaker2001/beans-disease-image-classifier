import argparse

from src.config import load_config
from src.train import train_pipeline

import os
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_pipeline(cfg)


if __name__ == "__main__":
    main()