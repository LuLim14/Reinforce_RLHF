import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
import datasets
import torch.nn.functional as F

from typing import Any
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from Config import Configs


def sft_training(model: Any, tokenizer: Any, chosen_train_dataset: Dataset, chosen_validation_dataset: Dataset,
              config: Configs) -> None:

    training_arg = SFTConfig(
        max_seq_length=config.max_length_sft,
        output_dir=config.path_to_checkpoints_sft_model,
        learning_rate=config.learning_rate_sft,
        num_train_epochs=config.num_train_epochs_sft,
        fp16=True
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arg,
        train_dataset=chosen_train_dataset,
        eval_dataset=chosen_validation_dataset,
        dataset_batch_size=config.dataset_batch_size_sft,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(os.path.join(config.path_to_checkpoints_sft_model, 'sft_model_result'))

