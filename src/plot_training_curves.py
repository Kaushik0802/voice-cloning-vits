import os
import pandas as pd
import matplotlib.pyplot as plt
import yaml

def plot_curves(train_log_path, val_log_path, output_dir):
    train_loss = pd.read_csv(train_log_path, header=None)
    val_loss = pd.read_csv(val_log_path, header=None)

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss[0], train_loss[1], label="Train Loss")
    plt.plot(val_loss[0], val_loss[1], label="Validation Loss")

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot at {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logs_dir = config["project"]["logs_dir"]

    train_log_path = os.path.join(logs_dir, "training_loss.csv")
    val_log_path = os.path.join(logs_dir, "validation_loss.csv")

    plot_curves(train_log_path, val_log_path, logs_dir)
    print("Plotting training curves...")