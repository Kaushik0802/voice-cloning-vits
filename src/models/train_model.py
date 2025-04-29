import argparse
import yaml
from src.models.fine_tuner import FineTuner

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    fine_tuner = FineTuner(config)
    fine_tuner.fine_tune()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()
    main(args.config)
