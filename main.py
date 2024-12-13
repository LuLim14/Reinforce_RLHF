import argparse
import os
import random
import gc
import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
import datasets
import torch.nn.functional as F

from collections import deque
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Any, Optional, Union
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig, RewardTrainer, RewardConfig
from transformers.utils import PaddingStrategy
from sft_train import sft_training
from reward_train import build_dataset_for_reward_model, reward_training
from Config import Configs, seed_everything, init_wandb
from reinforce import reinforce_with_baseline, get_reward_on_validation


def build_dataset(dataset_name: str) -> Union[Dataset, Dataset, Dataset, Dataset]:
    train_dataset = load_dataset(dataset_name, split="train")
    validation_dataset = load_dataset(dataset_name, split="validation")

    chosen_train_dataset = train_dataset.select(range(0, len(train_dataset), 2))
    chosen_validation_dataset = validation_dataset.select(range(0, len(validation_dataset), 2))

    chosen_train_dataset = chosen_train_dataset.select_columns(['response'])
    chosen_train_dataset = chosen_train_dataset.rename_column('response', 'text')

    chosen_validation_dataset = chosen_validation_dataset.select_columns(['response'])
    chosen_validation_dataset = chosen_validation_dataset.rename_column('response', 'text')

    return chosen_train_dataset, chosen_validation_dataset, train_dataset, validation_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforce project')

    parser.add_argument('--use_wandb', type=bool, required=True, help='True if you want to use wandb')
    parser.add_argument('--path_to_checkpoints_sft_model', type=str, required=True,
                        help='Path to sft model checkpoints')
    parser.add_argument('--path_to_checkpoints_reward_model', type=str, required=True,
                        help='Path to reward model checkpoints')

    args = parser.parse_args()
    Configs.use_wandb = args.use_wandb
    Configs.path_to_checkpoints_sft_model = args.path_to_checkpoints_sft_model
    Configs.path_to_checkpoints_reward_model = args.path_to_checkpoints_reward_model

    seed_everything()
    if Configs.use_wandb:
        init_wandb()

    chosen_train_dataset, chosen_validation_dataset, train_dataset, validation_dataset = build_dataset(Configs.dataset_name)

    model = AutoModelForCausalLM.from_pretrained(Configs.model_name)
    tokenizer = AutoTokenizer.from_pretrained(Configs.model_name)

    sft_training(model, tokenizer, chosen_train_dataset, chosen_validation_dataset, Configs)

    train_reward_dataset, validation_reward_dataset = build_dataset_for_reward_model(train_dataset, validation_dataset)

    reward_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(Configs.path_to_checkpoints_sft_model, 'sft_model_result'),
                                                                      num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(Configs.path_to_checkpoints_sft_model, 'sft_model_result'))

    reward_training(reward_model, tokenizer, train_reward_dataset, validation_reward_dataset, Configs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sft_model = AutoModelForCausalLM.from_pretrained(os.path.join(Configs.path_to_checkpoints_sft_model, 'sft_model_result')).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(Configs.path_to_checkpoints_reward_model, 'reward_model_result')).to(device)
    policy_model = AutoModelForCausalLM.from_pretrained(Configs.model_name).to(device)


    train_rl_dataset = chosen_train_dataset
    train_rl_dataloader = DataLoader(train_rl_dataset, batch_size=2, shuffle=True, num_workers=4)

    validation_rl_dataset = chosen_validation_dataset
    validation_rl_dataloader = DataLoader(validation_rl_dataset, batch_size=2, shuffle=False, num_workers=4)

    rewards_aggregated, all_rewards, policy_model = reinforce_with_baseline(sft_model, tokenizer, reward_model,
                                                                            train_rl_dataloader,
                                                                            validation_rl_dataloader,
                                                                            num_epochs=100)

    sft_reward, texts_for_reward_sft = get_reward_on_validation(sft_model, reward_model, tokenizer,
                                                                validation_rl_dataloader)
    policy_reward, texts_for_reward_policy = get_reward_on_validation(policy_model, reward_model, tokenizer,
                                                                      validation_rl_dataloader)

    print(f'Mean reward sft: {torch.sum(sft_reward.detach().cpu().mean(dim=0)).item()}')
    print(f'Mean reward sft: {torch.sum(policy_reward.detach().cpu().mean(dim=0)).item()}')

