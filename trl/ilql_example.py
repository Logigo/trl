# imports
import torch
from transformers import GPT2Tokenizer
from trl.gpt2 import GPT2HeadWithQValueModel, respond_to_batch
from trl.ilql import ILQLTrainer

# get models
gpt2_model = GPT2HeadWithQValueModel.from_pretrained('gpt2')
gpt2_model_ref = GPT2HeadWithQValueModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# initialize trainer
ilql_config = {'batch_size': 1, 'forward_batch_size': 1}
ilql_trainer = ILQLTrainer(gpt2_model, gpt2_tokenizer, **ilql_config)

# encode a query
query_txt = "This morning I went to the "
query_tensor = gpt2_tokenizer.encode(query_txt, return_tensors="pt")

# get model response
response_tensor  = respond_to_batch(gpt2_model, query_tensor)
response_txt = gpt2_tokenizer.decode(response_tensor[0,:])

# define a reward for response
# (this could be any reward such as human feedback or output from another model)
# TODO: What should this be? The +ve movie reviews used BERT. 
reward = [torch.tensor(1.0)]

# train model with ppo
train_stats = ilql_trainer.step([query_tensor[0]], [response_tensor[0]], reward)