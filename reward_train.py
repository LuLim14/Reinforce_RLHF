import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
import datasets
import torch.nn.functional as F

from typing import Any, Union
from trl import RewardTrainer, RewardConfig
from datasets import Dataset
from Config import Configs


def build_dataset_for_reward_model(train_dataset: Dataset, validation_dataset: Dataset) -> Union[Dataset, Dataset]:
    train_reward_dataset = {
        'prompt': [],
        'chosen': [],
        'rejected': []
    }

    validation_reward_dataset = {
        'prompt': [],
        'chosen': [],
        'rejected': []
    }

    for i, row in enumerate(train_dataset):
        if i % 2 == 0:
            train_reward_dataset['prompt'].append(row['prompt'])
            train_reward_dataset['chosen'].append(row['response'])
        else:
            train_reward_dataset['rejected'].append(row['response'])

    train_reward_dataset = pd.DataFrame(train_reward_dataset)
    train_reward_dataset = datasets.Dataset.from_pandas(train_reward_dataset)

    for i, row in enumerate(validation_dataset):
        if i % 2 == 0:
            validation_reward_dataset['prompt'].append(row['prompt'])
            validation_reward_dataset['chosen'].append(row['response'])
        else:
            validation_reward_dataset['rejected'].append(row['response'])

    validation_reward_dataset = pd.DataFrame(validation_reward_dataset)
    validation_reward_dataset = datasets.Dataset.from_pandas(validation_reward_dataset)
    return train_reward_dataset, validation_reward_dataset


def reward_training(model: Any, tokenizer: Any, train_reward_dataset: Dataset, validation_reward_dataset: Dataset,
                    config: Configs) -> None:

    reward_training_args = RewardConfig(
        output_dir=config.path_to_checkpoints_reward_model,
        max_length=config.max_length_reward,
        learning_rate=config.learning_rate_reward,
        num_train_epochs=config.num_train_epochs_reward,
        per_device_train_batch_size=config.dataset_batch_size_reward,
        per_device_eval_batch_size=config.dataset_batch_size_reward,
        fp16=True
    )

    reward_trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_training_args,
        train_dataset=train_reward_dataset,
        eval_dataset=validation_reward_dataset,
    )
    reward_trainer.train()
    reward_trainer.save_model(os.path.join(config.path_to_checkpoints_reward_model, 'reward_model_result'))

