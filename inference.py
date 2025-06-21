import argparse

import dotenv
from sklearn.metrics import accuracy_score

from configs import Config
from data_loaders import build_data_loader
from models import load_model_and_tokenizer, predict
from utils import set_seed


def inference(config: Config, validation: bool = False):
    model, tokenizer = load_model_and_tokenizer(config.inference.model_path, config)

    if validation:
        data_loader = build_data_loader("validation", tokenizer, config)
    else:
        data_loader = build_data_loader("inference", tokenizer, config)

    prediction = predict(model, tokenizer, data_loader.dataset)

    if validation:
        prediction.to_csv(config.train.valid_output_path, index=False)
        accuracy = accuracy_score(prediction["label"].astype(str), prediction["answer"].astype(str))
        print("\nFinal Validation results:")
        print(f"Accuracy: {accuracy:4f}")
    else:
        prediction.to_csv(config.inference.output_path, index=False)


if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, default="config.yaml")
    parser.add_argument("-v", "--validation", action="store_true")
    args = parser.parse_args()

    try:
        config = Config(args.config_file)
    except FileNotFoundError:
        print(f"Config file not found: {args.config_file}")
        print("Run with default config: config.yaml\n")
        config = Config()

    set_seed(config.common.seed)

    inference(config, validation=args.validation)
