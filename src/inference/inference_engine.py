# src/inference/inference_engine.py

import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt
from TTS.api import TTS
from src.utils.logger import get_logger

logger = get_logger()

class InferenceEngine:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = config["model"].get("restore_path")
        model_name = config["model"]["base_model"]

        self.tts = TTS(model_name=model_name, progress_bar=True).to(self.device)

        if model_path and os.path.exists(model_path):
            self.tts.load_checkpoint(model_path)
            logger.info(f"Loaded checkpoint from {model_path}")
        else:
            logger.warning("Checkpoint not found. Using pretrained base model.")

    def synthesize(self, text=None):
        if not text:
            text = self.config["inference"]["text_input"]

        out_path = self.config["inference"]["output_audio_path"]
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        wav = self.tts.tts(text)
        sf.write(out_path, wav, samplerate=self.config["dataset"]["sampling_rate"])
        logger.info(f"Saved audio at {out_path}")

        self._plot_waveform_and_spectrogram(wav, out_dir)

    def _plot_waveform_and_spectrogram(self, wav, out_dir):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        axs[0].plot(wav)
        axs[0].set_title("Waveform")

        axs[1].specgram(wav, Fs=self.config["dataset"]["sampling_rate"])
        axs[1].set_title("Spectrogram")

        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, "audio_plot.png"))
        logger.info("Saved waveform and spectrogram.")
