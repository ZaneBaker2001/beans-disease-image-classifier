import argparse
import json

from src.inference import predict_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--serving-config", required=True, type=str)
    args = parser.parse_args()

    result = predict_image(
        image_path=args.image,
        model_path=args.model,
        serving_config_path=args.serving_config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()