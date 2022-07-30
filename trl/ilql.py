# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02-ppo.ipynb (unless otherwise specified).

__all__ = ['AdaptiveKLController', 'FixedKLController', 'ILQLTrainer']

# Cell
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import torch
import collections
import time
import random
from typing import List
from transformers import DataCollatorForLanguageModeling
from trl.gpt2_ilql import CausalLMOutputWithCrossAttentions, GPT2HeadWithQValueModel

from .core import (   flatten_dict,
                      stats_to_np,
                      stack_dicts,
                      WANDB_PADDING)


torch.autograd.set_detect_anomaly(True)
# Cell
# 1. ILQLTrainer gets initialized with its parameters, a tokenizer, and a model (M).
# 2. The tokenizer is given a query (sentence) to encode into a Tensor Q_T.
# 3. The model M responds to the encoded query Q_T, and the tokenizer decodes the response R_T.
# 4. There is a reward r. Somehow. 
# 5. The ILQLTrainer calls Step on the query and response tensors (Q_T and R_T) and reward r.
# 6. Step(Q_T, R_T, r) -> Stats:
#   a. Runs batched_forward_pass(Q_T, R_T) to get logprobs, ref_logprobs, and values
#       a.i. For each iteration of size BS/FBS, runs the model M on input of [query, response].
#           to return logits, and values. Runs [query, response] through reference model M_r to return
#              reference logits. TODO: Should this return logits, values, and q, for ILQL?
#       a.ii. Logits and reference logits are turned into logprobs and ref_logprobs. Returns
#           lists of logprobs, ref_logprobs, and values
#   b. Runs compute_rewards(scores=r, logprobs, ref_logprobs) to get rewards, non_score_rewards
#       b.i. for each score (aka r), logprob, ref_logprob, computes KL Divergence as logprob - ref_logprob.
#           Math wise, subtraction of logs is log of their division. It then scales the KL Divergence,
#           and scores it as a non-score-reward and an identical value + score (r) as a reward.
#       b.ii. It then returns the rewards and non_score_rewards            
#   c. For each epoch: --> For each shuffled batch:
#       c.i. Calls train_minibatch(logprob[idx], values[idx], rewards[idx], queries[idx], responses[idx], [query, response])
#           c.i.1. For the logprob, value, reward, query, response, and original model input ([query, response] pair):
#           c.i.2. Calculates loss and runs propagates loss backwards, optimizer takes a step.
#           c.i.3. Loss Calculation(old_logprobs, values, rewards, query(unused), response, model_input)
# 

class ILQLTrainer:
    """
    The ILQLTrainer uses Implicit Language Q Learning to optimise language models.
    """

    default_params = {
        "lr": 1e-4, # From the paper, A.3
        "batch_size": 256,
        "forward_batch_size": 256,
        "visual_dialogue_batch_size": 64,
        "reddit_batch_size": 32,
        "ilql_epochs": 4,
        "gamma": 0.99, # From the paper, A.3
        "alpha": 0.1,
        "tau": 0.7,
        "polyak_decay": 0.005,
        'target_update_frequency': 10, # TODO: Find it from the paper, I pulled this number out of nowhere. 
    }

    def __init__(self, model, tokenizer, **ilql_params):
        """
        Initialize ILQLTrainer.

        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            tokenizer (tokenizer): Hugging Face tokenizer
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ilql_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'alpha' (float): Alpha parameter for CQL loss term, default: 0.1
                TODO: Add lambda_Q, lambda_V

        """
        # NOTE: Ilql params are processed here but can have implications in MLP heads' dimensions inside of GPT2, aka, self.model
        self.ilql_params = self.default_params
        self.ilql_params.update(ilql_params)

        self.model: GPT2HeadWithQValueModel = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        self.optimizer = Adam(model.parameters(), lr=self.ilql_params['lr'])
        self.step_count = 0


    def step(self, queries, responses, scores):
        """
        Run a ILQL optimisation step.

        args:
            queries (List): List of tensors containing the encoded queries, shape [query_length]
            responses (List): List of tensors containing the encoded responses, shape [response_length]
            scores (List): tensor containing the scores, shape [batch_size]

        returns:
            train_stats (dict): a summary of the training statistics
        """
        bs = self.ilql_params['batch_size']
        assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()        

        rewards = scores.detach()
        timing['time/ilql/compute_rewards'] = time.time()-t0

        t = time.time()
        all_stats = []      
        input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(queries, responses)])["input_ids"]
        timing['time/ilql/collate_data'] = time.time()-t

        idxs = list(range(bs))
        for _ in range(self.ilql_params['ilql_epochs']):
            random.shuffle(idxs)
            # input_ids is of batch_size
            for i, input_id in enumerate(input_ids):
                idx = idxs[i]
                query = queries[idx]
                response = responses[idx]
                reward = rewards[idx]
                train_stats = self.train_input(query, response, reward, input_id)
                all_stats.append(train_stats)
        timing['time/ilql/optimize_step'] = time.time()-t

        # Update targets
        if self.step_count + 1 % self.ilql_params['target_update_frequency'] == 0:
            t = time.time()
            # Uses EMA to update target q network
            self.model.soft_update_targets(alpha=self.ilql_params['polyak_decay'])
            train_stats['time/ilql/target_update'] = time.time()-t

        t = time.time()
        stats = self.stack_dicts(all_stats)
        # TODO: Update record_step_stats with the right values
        # stats = self.record_step_stats(scores=scores, train_stats=train_stats)
        # stats = stats_to_np(stats)
        timing['time/ilql/calc_stats'] = time.time()-t
        timing['time/ilql/total'] = time.time()-t0
        train_stats.update(timing)
        stats.update(train_stats)
        self.step_count += 1
        print(f'Step: {self.step_count}')

        return train_stats

    # TODO: This is my own version of stats_dicts that is not applied to tensor values, but float values 
    #  Uses recursion for nested dictionaries
    def stack_dicts(self, stats: List[dict] or dict) -> dict:
        results = {} # Assumes each dict in the list has the same keys
        stats_keys = stats[0].keys() if type(stats) == list else stats.keys()
        # [1, {x y}, 2]
        for key in stats_keys:
            list_of_values = [self.stack_dicts(_dict[key]) if type(_dict[key]) == dict else _dict[key] for _dict in stats]
            results[key] = list_of_values
        return results 

    def train_input(self, queries, responses, rewards, input_id):
        # Compute model output
        stats = {}
        t0 = time.time()

        lmo: CausalLMOutputWithCrossAttentions = self.model(input_id)
        logits, _, v, q1, q2, target_q = lmo.logits.squeeze(0), lmo.attentions, lmo.value.squeeze(0), lmo.qs[0].squeeze(0), lmo.qs[1].squeeze(0), lmo.target_qs[0].squeeze(0)
        stats['forward_pass'] = time.time()-t0
    
        # Compute ILQL Loss
        t = time.time()
        sequence_len = responses.shape[0]
        loss, loss_step_stats = self.ilql_loss(v, [q1, q2], target_q, rewards, sequence_len)
        # TODO: This is just a syntactically correct placeholder to update stats
        stats['loss'] = {'total_time': time.time()-t, **loss_step_stats}
        # Backward propagate batch
        t = time.time()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        stats['backprop'] = time.time()-t
        return flatten_dict(stats)


    def batched_forward_pass(self, queries, responses):
        """Calculate model outputs in multiple batches."""
        bs = self.ilql_params['batch_size']
        fbs = self.ilql_params['forward_batch_size']
        all_logprobs = []
        all_values = []
        all_q1 = []
        all_q2 = []
        all_target_q = []
        all_attn_masks = []
        # NOTE: Should this exist? I am thinking to have it so I can pass next states to compute target_q
        all_states = []

        for i in range(int(bs/fbs)):
            query_batch = queries[i*fbs:(i+1)*fbs]
            response_batch = responses[i*fbs:(i+1)*fbs]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])["input_ids"]
            # LM OutputW
            # TODO: No grad the forward pass?
            with torch.no_grad():
                print(f'\tBatched Forward Pass: Running forward pass')
                lmo: CausalLMOutputWithCrossAttentions = self.model(input_ids)
            logits, attn_masks, v, q1, q2, target_q = lmo.logits, lmo.attentions, lmo.value, lmo.qs[0], lmo.qs[1], lmo.target_qs[0]
            # This type of debugging should have me sent straight to hell
            # q = torch.minimum(q1, q2)
            # target_q = torch.minimum(target_q1, target_q2)
            # TODO: What is happening with the indices here?
            logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
            # TODO: Inspect fbs and bs/fbs etc. to figure out indexing
            print(f'\tBatched Forward Pass; Updating lists')

            for j in range(fbs): 
                start = len(query_batch[j])-1
                end = len(query_batch[j]) + len(response_batch[j])-1
                # TODO: Is this right? Who knows! Where is the indexing from? Is this what's causing the modifications?
                all_q1.append(q1[j, start-1:end-1])
                all_q2.append(q2[j, start-1:end-1])
                all_target_q.append(target_q[j, start-1:end-1])
                all_values.append(v[j, start-1:end-1])
                # TODO: Attn masks are none
                # all_attn_masks.append(attn_masks[j, start-1:end-1]) # Is indexing right? For any of these?
                all_logprobs.append(logprobs[j, start:end])
        print(f'\tBatched Forward Pass updated all_q, all_target_q, all_values, and all_logprobs')
        return all_logprobs, all_values, all_q1, all_q2, all_target_q, all_attn_masks


        pass


    def L_QV(self, value, next_value, q, q_hat, reward):
        t = time.time()
        def _L_doubleQ(q1, q2, _qt):
            q1_loss = (_qt - q1) ** 2
            q2_loss = (_qt - q2) ** 2
            return q1_loss + q2_loss

        # According to https://github.com/Sea-Snell/Implicit-Language-Q-Learning/blob/13e3d58ee27527a0c819c92702d322a829211540/src/models/iql_model.py#L356
        next_value = next_value.detach()
        gamma = self.ilql_params['gamma']
        q_target = reward + (gamma * next_value) # Not to be confused with the result of the target_Q transformer. 
        q1, q2 = q[0], q[1]
        L_Q = _L_doubleQ(q1, q2, q_target)

        # Expectile Regression: - I wrote this to be as readable as possible, not as fast as possible
        q_hat = q_hat.detach()
        u = (value - q_hat) # Q_hat - V, but flipped
        ones, zeros = torch.ones(u.shape), torch.zeros(u.shape)
        fancy_one_symbol = torch.where(u > 0, ones, zeros) # Usually, this is 1 if u < 0, but we flipped it cause we are minimizing and not maximizing, therefore u is a negative number
        tau = self.ilql_params['tau']
        expectile = torch.abs(tau - fancy_one_symbol) * (u ** 2)
        L_V = expectile 

        expectation_L_Q = L_Q.mean()
        expectation_L_V = L_V.mean()
        # TODO: Should I record E[L_Q] and E[L_V] here? NOTE: I return E[L_V] so I can record both E[L_V] and E[Q_V] in ilql_loss
        return expectation_L_Q + expectation_L_V, expectation_L_V, time.time()-t


    def L_CQL(self, q):
        t = time.time()
        q1, q2 = q[0], q[1]
        numerator1, numerator2 = torch.exp(q1), torch.exp(q2) 
        denominator1, denominator2 = torch.sum(torch.exp(q1)), torch.sum(torch.exp(q2))
        cql1, cql2 = torch.log(torch.divide(numerator1, denominator1)), torch.log(torch.divide(numerator2, denominator2))
        cql = cql1 + cql2
        # We return CQL in expectation
        return cql.mean(), time.time()-t


    def ilql_loss(self, values, double_qs, double_target_qs, reward, sequence_length):
        # TODO: Rework this to work with stack_dicts - key difference with this loss and ppo is that this is applied to multiple words
        stats = {}
        total_qv_loss = 0.
        total_cql_loss = 0.
        total_v_loss = 0.
        total_qv_time = 0.
        total_cql_time = 0.

        # TODO: Might need something other than stack_dicts for a loss function like this - check ppo.py vs this to see
        for i in range(sequence_length): 
            next_value = values[i + 1].detach() if i + 1 < sequence_length else torch.tensor(0., requires_grad=False)
            value, double_q, double_target_q = values[i], [double_qs[0][i], double_qs[1][i]], double_target_qs[i]
            qv_loss, v_loss, qv_time = self.L_QV(value, next_value, double_q, double_target_q, reward)
            cql_loss, cql_time = self.L_CQL(double_q)
            # Update totals for more granular logging
            total_qv_loss += qv_loss
            total_cql_loss += cql_loss
            total_v_loss += v_loss
            total_qv_time += qv_time
            total_cql_time += cql_time
            # Accumulates time spent on qv and cql loss terms in the stats dictionary 
        # TODO: What does flatten_dict do haha 
        total_loss = total_qv_loss + (self.ilql_params['alpha'] * cql_loss)
        stats = dict(
            loss=dict(total=total_loss, q_loss=total_qv_loss-total_v_loss, v_loss=total_v_loss, 
            qv_loss=total_qv_loss, cql_loss=total_cql_loss),
        )
        return total_loss, stats

    # TODO: Repurpose for ILQL
    def record_step_stats(self, **data):
        """Record training step statistics."""
        stats = {}
        for k, v in data['train_stats'].items():
            stats[f'ilql/{k}'] = torch.mean(v, axis=0)
            
        return stats

# NOTE: This replaces the same function from core.py, and specifies a dimension, because I am getting logprobs at the per-input level
#  since there is no batched forward pass for ILQL like in PPO. 
def logprobs_from_logits(logits, labels, dim=2):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=dim)
    logpy = torch.gather(logp, dim, labels.unsqueeze(dim)).squeeze(-1)
    return logpy