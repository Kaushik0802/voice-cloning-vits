# Voice Cloning with VITS (Fine-Tuning on LJSpeech)

This project fine-tunes a VITS-based Text-to-Speech model to clone a voice using the LJSpeech dataset. It enables fast and lightweight fine-tuning to generate high-quality synthesized speech.

---

## Project Structure
final_voice_cloning_project/
├── src/
│   ├── models/
│   ├── data/
│   ├── inference/
│   ├── utils/
│   └── config/
├── notebooks/
│   ├── voice_cloning_colab.ipynb
│   ├── starter_inference_colab.ipynb
├── experiments/   # (auto-generated)
├── ethical_audit/
├── outputs/       # (empty initially)
├── README.md
├── requirements.txt
├── environment.yml
├── Dockerfile
├── final_report.pdf


---

## How to Run (Colab Steps)

1. Upload `final_voice_cloning_project.zip` and unzip.

2. Install dependencies:
   !pip install -r requirements.txt

3. Prepare dataset:
!python src/data/prepare_dataset.py --config src/config/config.yaml

4. Fine-tune model:
!python src/models/train_model.py --config src/config/config.yaml

5. Plot training curves:
!python src/plot_training_curves.py --config src/config/config.yaml

6. Run inference:
!python src/inference/run_inference.py --config src/config/config.yaml

## Technical Details
Model: VITS (from Coqui TTS)

Dataset: LJSpeech-1.1

Training: 500 epochs, cosine scheduler, Adam optimizer

Loss Plotting: Smooth training/validation loss curves

Inference: Text input → cloned audio output (.wav)

## Ethical Considerations
See ethical_audit/bias_and_privacy_analysis.md for important notes on responsible use, data privacy, and potential bias.

## Results
Training loss and validation loss plotted.

Synthesized speech output sounds realistic.

Spectrograms and waveform plots generated.

## Requirements
Python 3.8+

torch

TTS (Coqui.ai)

pandas

matplotlib

seaborn

pyyaml

soundfile

tqdm

Install all via:
pip install -r requirements.txt

Credits
LJSpeech Dataset: LJ Speech Dataset

Coqui TTS: Coqui.ai TTS