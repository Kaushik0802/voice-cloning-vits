# src/data/dataset_manager.py

import os
from torch.utils.data import Dataset

class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.metadata = [line.strip().split("|") for line in lines]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_file, text, *_ = self.metadata[idx]
        return {"wav_path": wav_file, "text": text}
