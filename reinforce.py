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
from tqdm.auto import tqdm
from typing import Union, Any


def reinforce_with_baseline(policy, tokenizer, reward_model, train_dataloader,
                            validation_dataloader, num_epochs, gamma=0.99) -> Union[list, list, Any]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy.train()
    reward_model.eval()

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    all_rewards = []
    rewards_aggregated = []
    mx_reward = 0.0
    for episode in tqdm(range(num_epochs)):
        log_probs = []
        rewards = []

        for j, item in tqdm(enumerate(train_dataloader)):
            tokenized = tokenizer(text=item['text'], truncation=True, padding='max_length',
                                  max_length=512, return_tensors='pt').to(device)
            input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
            response = policy.generate(input_ids=input_ids,
                                       attention_mask=attention_mask, max_new_tokens=20,
                                       top_p=0.95,
                                       do_sample=True,
                                       num_return_sequences=1).cpu()

            torch.cuda.empty_cache()
            gc.collect()
            texts_for_reward = []
            texts_for_reward = tokenizer.batch_decode(response, skip_special_tokens=True)

            ##### get reward:
            texts_for_reward = ["Question: " + question_text + "\n\nAnswer: " + answer_text for
                                question_text, answer_text in zip(item['text'], texts_for_reward)]
            response_tokenized = tokenizer(texts_for_reward, truncation=True, padding='max_length', max_length=512,
                                           return_tensors='pt').to(device)
            response_input_ids, response_attention_mask = response_tokenized['input_ids'], response_tokenized['attention_mask']
            with torch.no_grad():
                reward = reward_model(input_ids=response_input_ids,
                                      attention_mask=response_attention_mask)
            torch.cuda.empty_cache()
            gc.collect()
            rewards.append(reward.logits)
            all_rewards.append(reward.logits)
            ################

            ##### get log_prob:
            outputs = policy(input_ids=response_input_ids, attention_mask=response_attention_mask)
            last_token_logits = outputs.logits[:, -1, :]
            log_prob = F.log_softmax(last_token_logits, dim=-1)
            log_probs.append(log_prob)
            ##################
            torch.cuda.empty_cache()
            gc.collect()

        n = len(rewards)
        G = 0
        discounted_rewards = deque()
        for t in reversed(range(n)):
            G = rewards[t] + gamma * G
            discounted_rewards.appendleft(G)

        del rewards
        discounted_rewards = torch.stack(list(discounted_rewards))
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        baseline = torch.mean(torch.stack(all_rewards), dim=0).float()
        for discount_reward, log_prob in zip(discounted_rewards, log_probs):
            policy_loss.append(-(discount_reward - baseline) * log_prob)
        policy_loss = torch.cat(policy_loss).sum()
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        print(
            f'Episode: {episode}, loss: {policy_loss}, sum of rewards in batch: {torch.sum(discounted_rewards.detach().cpu().mean(dim=0))}')
        rewards_aggregated.append(torch.sum(discounted_rewards.detach().cpu().mean(dim=0)).item())
        wandb.log({
            'episode': episode,
            'loss': policy_loss,
            'sum_mean_discount_reward_batch': torch.sum(discounted_rewards.detach().cpu().mean(dim=0)).item()
        })
        if torch.sum(discounted_rewards.detach().cpu().mean(dim=0)).item() > mx_reward:
            mx_reward = torch.sum(discounted_rewards.detach().cpu().mean(dim=0)).item()
            policy.save_pretrained(os.path.join('/kaggle/working/', 'temp_policy'))
        torch.cuda.empty_cache()
        gc.collect()

    policy.save_pretrained(os.path.join('/kaggle/working/', 'last_policy'))
    return rewards_aggregated, all_rewards, policy


def get_reward_on_validation(model, reward_model, tokenizer, validation_dataloader, gamma=0.99) -> Union[deque, list]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    rewards = []
    for j, item in enumerate(validation_dataloader):
        tokenized = tokenizer(text=item['text'], truncation=True, padding='max_length',
                              max_length=512, return_tensors='pt').to(device)
        input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        response = model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask, max_new_tokens=20,
                                  top_p=0.95,
                                  do_sample=True,
                                  num_return_sequences=1).cpu()

        torch.cuda.empty_cache()
        gc.collect()

        texts_for_reward = []
        for response_item in response:
            decode = [tokenizer.decode(token_ids=token_id, skip_special_tokens=True) for token_id in response_item]
            decoded_text = ' '.join(decode)
            if j == 0:
                print(decoded_text)
            texts_for_reward.append(decoded_text)

        ##### get reward:
        texts_for_reward = [answer_text for question_text, answer_text in zip(item['text'], texts_for_reward)]
        response_tokenized = tokenizer(texts_for_reward, truncation=True, padding='max_length', max_length=512,
                                       return_tensors='pt').to(device)
        response_input_ids, response_attention_mask = response_tokenized['input_ids'], response_tokenized['attention_mask']
        with torch.no_grad():
            reward = reward_model(input_ids=response_input_ids,
                                  attention_mask=response_attention_mask)
        torch.cuda.empty_cache()
        gc.collect()

        rewards.append(reward.logits)

        if j == 6:
            break

    n = len(rewards)
    G = 0
    discounted_rewards = deque()
    for t in reversed(range(n)):
        G = rewards[t] + gamma * G
        discounted_rewards.appendleft(G)

    discounted_rewards = torch.stack(list(discounted_rewards))
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    return discounted_rewards, texts_for_reward

