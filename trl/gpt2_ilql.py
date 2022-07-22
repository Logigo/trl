# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01-gpt2-with-value-head.ipynb (unless otherwise specified).

__all__ = ['CausalLMOutputWithCrossAttentions', 'ValueHead', 'GPT2HeadWithValueModel', 'respond_to_batch']

# Cell
from mimetypes import init
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from transformers.modeling_outputs import ModelOutput
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional, Tuple
import copy

# Cell
@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None
    qs: Optional[Tuple[torch.FloatTensor]] = None
    target_qs: Optional[Tuple[torch.FloatTensor]] = None

# Cell
# Cell
# TODO: This is the Value Function Transformer (I assume). In the ILQL paper, it says:
# Our value function transformer has three MLP heads: two independently initialized and trained Q heads and one V head. 
'''
It also says:

We run all of our experiments on GPT-2 small transformer architectures,
    with the supervised learning policy on one transformer
    and Q and V heads on a separate one. 
    The target Q network is also on a separate transformer

So I need 3 transformers? What is a supervised learning policy?

'''

class MLPHead(nn.Module): 
    def __init__(self, config, output_size=1) -> None:
        super().__init__()
        # TODO: Should I detach head ever?
        self.hidden_dimension = config.n_embd
        self.linear_1 = nn.Linear(self.hidden_dimension, self.hidden_dimension*2)
        self.non_linearity = nn.ReLU()
        self.linear_2 = nn.Linear(self.hidden_dimension*2, output_size)
        # TODO: Whether we use num_tokens() or 1 as an output dimension depends on if it is PerToken (1) or PerUtterance (num_tokens())
    def forward(self, x):
        x = self.linear_1(x)
        x = self.non_linearity(x)
        x = self.linear_2(x)
        return x

class GPT2HeadWithQValueModel(GPT2PreTrainedModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    # TODO: How do I add arguments to config from outside the 
    def __init__(self, config, utterance=False):
        super().__init__(config)
        config.num_labels = 1
        # n_embd = 768
        # vocab_size = 50257,
        # eos and bos token id = 50256
        self.num_tokens = config.vocab_size if not utterance else 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Q Head
        self.q1 = MLPHead(config, output_size=self.num_tokens)
        self.q2 = MLPHead(config, output_size=self.num_tokens)

        # Target transformer is initialized as a copy of self.transformer
        self.target_transformer = copy.deepcopy(self.transformer)
        # TODO: "Our target Q networks are Polyak-averaged with decay factor 0.005 for both the transformer and the Q function head"
        #   ^ What does this mean?
        self.target_q1 = MLPHead(config, output_size=self.num_tokens)
        self.target_q2 = MLPHead(config, output_size=self.num_tokens)

        self.v_head = MLPHead(config, output_size=1)


        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    # TODO: What does this mean? Is this just detatching it from propagating gradients?
    def detach_value_head(self):
        self.v_head.detach_head = True

    def soft_update_targets(self, alpha):
        stats = {}
        def _copy_parameters_ema(target, copy):
            nonlocal stats, alpha
            for target_param, copy_param in zip(target.parameters(), copy.parameters()):
                ema_copy_param = (alpha * copy_param.data) + (1.0 - alpha)*target_param.data
                target_param.data.copy_(ema_copy_param)
        """
        for target_param, local_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(self.alpha*local_param.data + (1.0-self.alpha)*target_param.data)
        if self.double_q:
            for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(self.alpha*local_param.data + (1.0-self.alpha)*target_param.data)
        if self.lm_target is not None:
            for target_param, local_param in zip(self.lm_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.alpha*local_param.data + (1.0-self.alpha)*target_param.data)"""
        # TODO: Which one are we? We are double Q, and lm_target is not none (on a diff transformer).
        #  Refer to: https://github.com/Sea-Snell/Implicit-Language-Q-Learning/blob/13e3d58ee27527a0c819c92702d322a829211540/src/models/iql_model.py#L141
        # Copy target_q1<--q1
        tq1_stats = _copy_parameters_ema(self.target_q1, self.q1)
        # Copy target_q2<--q2
        tq2_stats = _copy_parameters_ema(self.target_q2, self.q2)
        # Copy target_transformer<--transformer
        tformer_stats = _copy_parameters_ema(self.target_transformer, self.transformer)
        return (tq1_stats, tq2_stats, tformer_stats)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss=None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0] # [1, len(input_ids), 50257], which indices are action and state hidden states?
        # h_dim = n_embd = 768
        # What does this mean below? What would state_idxs/action_idxs be?
        """state_hidden_states = torch.gather(input=hidden_states, dim=1, index=state_idxs.unsqueeze(2).repeat(1, 1, self.h_dim))"""
        state_hidden_states = torch.clone(hidden_states)
        """action_hidden_states = torch.gather(input=hidden_states, dim=1, index=action_idxs.unsqueeze(2).repeat(1, 1, self.h_dim))"""
        action_hidden_states = torch.clone(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        
        # A.3 from ILQL: "The target Q network is also on a separate transformer"
        # NOTE: Does the target Q take input_ids?
        with torch.no_grad():
            target_transformer_outputs = self.target_transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
        target_hidden_states = target_transformer_outputs[0]
        """action_target_hidden_states = torch.gather(input=target_hidden_states, dim=1, index=action_idxs.unsqueeze(2).repeat(1, 1, self.h_dim))"""
        action_target_hidden_states = torch.clone(target_hidden_states)

        value = self.v_head(state_hidden_states).squeeze(-1)
        # TODO: I just copied the line above. Most likely incorrect. 
        # Polyak averaged Q Heads
        # TODO: The Q should take action_hidden states, but the value should take the state hidden state. Why? Look at 287-291:
        # https://github.com/Sea-Snell/Implicit-Language-Q-Learning/blob/13e3d58ee27527a0c819c92702d322a829211540/src/models/iql_model.py#L287-L291
        q1 = self.q1(action_hidden_states).squeeze(-1)
        q2 = self.q2(action_hidden_states).squeeze(-1)
        # TODO: These should have no_grad() according to 
        #  https://github.com/Sea-Snell/Implicit-Language-Q-Learning/blob/13e3d58ee27527a0c819c92702d322a829211540/src/models/iql_model.py#L294
        with torch.no_grad():
            target_q1 = self.target_q1(action_target_hidden_states).squeeze(-1)
            target_q2 = self.target_q2(action_target_hidden_states).squeeze(-1)

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:] + (value,) + (q1,) + (q2,) + (target_q1,) + (target_q2,)
            return outputs

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states, 
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            value=value,
            qs=(q1, q2),
            target_qs=(target_q1, target_q2)
        )
    

# Cell

def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]