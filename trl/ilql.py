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

from transformers import DataCollatorForLanguageModeling

from .core import (logprobs_from_logits,
                      whiten,
                      clip_by_value,
                      entropy_from_logits,
                      flatten_dict,
                      average_torch_dicts,
                      stats_to_np,
                      stack_dicts,
                      add_suffix,
                      WANDB_PADDING)

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
        "forward_batch_size": 16,
        "visual_dialogue_batch_size": 64,
        "reddit_batch_size": 32,
        "ilql_epochs": 4,
        "gamma": 0.99, # From the paper, A.3
        "alpha": 0.1,
        "tau": 0.7,
        "polyak_decay": 0.005

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
        self.ilql_params = self.default_params
        self.ilql_params.update(ilql_params)

        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        self.optimizer = Adam(model.parameters(), lr=self.ppo_params['lr'])

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
        # TODO: Where does the GPT2HeadWithQValueModel tie into this?
        bs = self.ilql_params['batch_size']
        assert bs == len(queries), f"Batch size ({bs}) does not match number of examples ({len(queries)})"

        timing = dict()
        t0 = time.time()

        response_lengths = [len(r) for r in responses]

        t = time.time()
        # TODO: add next_v
        logprobs, ref_logprobs, v, q, target_q = self.batched_forward_pass(queries, responses)
        timing['time/ilql/forward_pass'] = time.time()-t

        t = time.time()
        # TODO: Should this even exist in ILQL? Don't we get rewards from BERT? What does this do?
        # rewards, non_score_reward = self.compute_rewards(scores, logprobs, ref_logprobs)
        # TODO: Since ILQL Rewards are just +ve word sentiments (without any KL divergence term), it is kept
        #  as is.
        rewards = scores
        timing['time/ilql/compute_rewards'] = time.time()-t

        t = time.time()
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.ilql_params['ilql_epochs']):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                # TODO: Change function arguments. 
                # Refer to https://github.dev/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/iql_trainer.py
                # I think this needs: 
                #  q_pred (detached): min(model.tq1(s, a), model.tq2(s, a)),
                #  q1_pred, q2_pred <-- self.q1(s, a)
                #  v: model.v(s),
                #  target_v: model.v(next_s) (detached) <-- TODO
                next_v = v[idx + 1] if idx < bs else 0.
                train_stats = self.train_minibatch(logprobs[idx].unsqueeze(0), v[idx].unsqueeze(0), next_v,
                                                    q[idx].unsqueeze(0), target_q[idx].unsqueeze[0], rewards[idx].unsqueeze(0),
                                                    queries[idx].unsqueeze(0), responses[idx].unsqueeze(0), 
                                                    torch.cat([queries[idx],responses[idx]]).unsqueeze(0))
                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
        train_stats['policy/advantages'] = torch.nan_to_num(train_stats['policy/advantages'], WANDB_PADDING)
        train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)

        # TODO: Update record_step_stats with the right values
        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward, train_stats=train_stats,
                                       kl_coef=self.kl_ctl.value)
        stats = stats_to_np(stats)
        timing['time/ilql/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ilql/total'] = time.time()-t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, queries, responses):
        """Calculate model outputs in multiple batches."""
        bs = self.ilql_params['batch_size']
        fbs = self.ilql_params['forward_batch_size']
        all_logprobs = []
        all_ref_logprobs = []
        all_states = [] # NOTE : TODO - If this is passed into the loss fn, where does V(s') get computed?
        # In the model, or in the loss fn?
        all_values = []
        all_q = []
        all_target_q = []

        for i in range(int(bs/fbs)):
            query_batch = queries[i*fbs:(i+1)*fbs]
            response_batch = responses[i*fbs:(i+1)*fbs]
            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])["input_ids"]
            with torch.no_grad():
                # TODO: q1 and q2 should have min taken
                logits, _, v, q1, q2, target_q1, target_q2 = self.model(input_ids)
                # TODO: Is this the right function?
                q = torch.minimum(q1, q2)
                target_q = torch.minimum(target_q1, target_q2)
                ref_logits, _, _ = self.ref_model(input_ids)
            # TODO: What is happening with the indices here?
            logprobs = logprobs_from_logits(logits[:,:-1,:], input_ids[:,1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:,:-1,:], input_ids[:,1:])
            for j in range(fbs):
                start = len(query_batch[j])-1
                end = len(query_batch[j]) + len(response_batch[j])-1
                # TODO: Is this right? Who knows!
                all_q.append(q[j, start-1:end-1])
                all_target_q.append(target_q[j, start-1:end-1])
                all_values.append(v[j, start-1:end-1])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])
        return all_logprobs, all_ref_logprobs, all_values, all_q, all_target_q


    # TODO: Arguments need to be changed. 
    def train_minibatch(self, logprob, value, next_value, q, target_q, reward, query, response, model_input):
        loss, train_stats = self.ilql_loss(logprob, value, next_value, q, target_q, reward, query, response, model_input)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return train_stats

    def L_QV(self, value, next_value, q, target_q, reward):
        # R(s, a) + V(s_i+1) - Q(s, a) <-- squared
        # +
        # Expectile loss(Q_hat(s, a) - V(s) )
        gamma = self.ilql_params['gamma']
        L_Q = (reward + (gamma * next_value) - q) ** 2
        # Expectile Regression: - I wrote this to be as readable as possible.
        u = (value - target_q) # Q_hat - V, but flipped
        fancy_one_symbol = 1. if u > 0 # Usually, this is 1 if u < 0, but we flipped it cause we are minimizing and not maximizing, therefore u is a negative number
        tau = self.ilql_params['tau']
        expectile = torch.abs(tau - fancy_one_symbol) * (u ** 2)
        L_V = expectile # In the IQL implementation, it is averaged with .mean(), but I think that's cause the loss
        # is applied on the whole batch at the same time (like, implementation wise), whereas the loss fn here takes 1 entry at a time
        return L_Q + L_V, {}

    def L_CQL(self, old_logprobs, values, rewards, query, response, model_input):
        """
        if self.double_q:
            q1, q2 = qs
            b, t, d = q1.shape
            return 
                (
                 (F.cross_entropy(q1.reshape(-1, d) / self.cql_temp, action_tokens.reshape(-1), reduction='none').reshape(b, t)
                 * (1 - terminals[:, :-1])) 
                 +
                 (F.cross_entropy(q2.reshape(-1, d) / self.cql_temp, action_tokens.reshape(-1), reduction='none').reshape(b, t)
                 * (1 - terminals[:, :-1]))
                 )
                 .sum() / max(n.item(), 1.0)
        I am not quite sure what the lines above me mean (https://github.com/Sea-Snell/Implicit-Language-Q-Learning/blob/13e3d58ee27527a0c819c92702d322a829211540/src/models/iql_model.py#L366-L369)
        But I am working on it
        """
        return None, {}

    def ilql_loss(self, old_logprob, value, next_value, q, target_q, rewards, query, response, model_input):
        qv_loss, qv_stats = self.L_QV(value, next_value, q, target_q, rewards)
        cql_loss, cql_stats = self.L_CQL(old_logprob, value, rewards, query, response, model_input)
        stats = {qv_stats, cql_stats} # TODO: What is this?
        return qv_loss + (self.ilql_params['alpha'] * cql_loss), flatten_dict(stats)

    def loss(self, old_logprobs, values, rewards, query, response, model_input):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        logits, _, vpred = self.model(model_input)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])

        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())

        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
                        advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
                     clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        )
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl_list = [logprobs-ref_logprobs for logprobs, ref_logprobs in zip(data['logprobs'], data['ref_logprobs'])]
        mean_kl = torch.mean(torch.stack([torch.sum(kl) for kl in kl_list]))
        mean_entropy = torch.mean(torch.stack([torch.sum(-log_probs) for log_probs in data['logprobs']]))
        mean_non_score_reward =torch.mean(torch.stack([torch.sum(non_score_reward) for non_score_reward in data['non_score_reward']]))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl_list,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats

