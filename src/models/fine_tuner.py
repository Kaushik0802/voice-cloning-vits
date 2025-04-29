import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.shared_configs import BaseDatasetConfig

class FineTuner:
    def __init__(self, config):
        self.config = config

        # Define dataset configuration
        dataset_config = BaseDatasetConfig(
            formatter="ljspeech",
            meta_file_train=config["dataset"]["meta_file_train"],
            meta_file_val=config["dataset"]["meta_file_val"],
            path=config["dataset"]["path"]
        )

        # Initialize model configuration
        self.model_config = VitsConfig(
            batch_size=config["training"]["batch_size"],
            eval_batch_size=config["training"]["batch_size"],
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=config["training"]["epochs"],
            text_cleaner="phoneme_cleaners",
            use_phonemes=True,
            phoneme_language="en-us",
            phoneme_cache_path=os.path.join(config["project"]["output_dir"], "phoneme_cache"),
            print_step=25,
            print_eval=False,
            mixed_precision=config["training"]["mixed_precision"],
            output_path=config["project"]["output_dir"],
            datasets=[dataset_config],
        )

        # Initialize Audio Processor
        self.ap = AudioProcessor.init_from_config(self.model_config)

        # Initialize Tokenizer
        self.tokenizer, self.model_config = TTSTokenizer.init_from_config(self.model_config)

        # Load training and evaluation samples
        self.train_samples, self.eval_samples = load_tts_samples(
            dataset_config,
            eval_split=True,
            eval_split_max_size=self.model_config.eval_split_max_size,
            eval_split_size=self.model_config.eval_split_size,
        )

        # Initialize the model
        self.model = Vits(self.model_config, self.ap, self.tokenizer)

        # Initialize the trainer
        self.trainer = Trainer(
            TrainerArgs(),
            self.model_config,
            config["project"]["output_dir"],
            model=self.model,
            train_samples=self.train_samples,
            eval_samples=self.eval_samples,
        )

    def fine_tune(self):
        self.trainer.fit()
