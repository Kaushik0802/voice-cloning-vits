# src/inference/run_inference.py

import argparse
import yaml
from src.inference.inference_engine import InferenceEngine
from src.utils.logger import setup_logging

def main(config_path, input_text=None):
    setup_logging()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    engine = InferenceEngine(config)
    engine.synthesize(input_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--text", required=False)
    args = parser.parse_args()
    main(args.config, args.text)
