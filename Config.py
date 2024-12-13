import os
import random
import torch
import numpy as np
import wandb

from typing import Optional


class Configs:
    use_wandb: Optional[bool] = False
    dataset_name: Optional[str] = 'nvidia/HelpSteer2'
    model_name: Optional[str] = 'facebook/opt-350m'

    path_to_checkpoints_sft_model: Optional[str] = './sft_model'
    max_length_sft: Optional[int] = 768
    learning_rate_sft: Optional[float] = 1e-4
    num_train_epochs_sft: Optional[int] = 1
    dataset_batch_size_sft: Optional[int] = 16

    path_to_checkpoints_reward_model: Optional[str] = './reward_model'
    max_length_reward: Optional[int] = 512
    learning_rate_reward: Optional[float] = 1e-5
    num_train_epochs_reward: Optional[int] = 1
    dataset_batch_size_reward: Optional[int] = 8


def seed_env(seed: int = 42) -> None:
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)


def seed_torch(seed: int = 42) -> None:
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def seed_everything() -> None:
  """Set seeds"""
  seed_torch()
  seed_env()


def init_wandb() -> None:
  wandb.init(project='Reinforce_RLHF', entity='lulim')

