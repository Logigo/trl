# imports
import torch
import numpy as np
from transformers import GPT2Tokenizer
from trl.gpt2_ilql import GPT2HeadWithQValueModel, respond_to_batch
from trl.ilql import ILQLTrainer
import wandb
from transformers import AutoTokenizer, pipeline
from trl.core import build_bert_batch_from_txt, listify_batch
from datasets import load_dataset
import time
from tqdm import tqdm

# NOTE: Straight up copied from https://github.com/lvwerra/trl/blob/master/nbs/04-gpt2-sentiment-ppo-training.ipynb
ilql_config = {'batch_size': 1, 'forward_batch_size': 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

wandb.init(name='run-1', project='ilql-gpt2-test', config=ilql_config, )

# load imdb with datasets
ds = load_dataset('imdb', split='train')
ds = ds.rename_columns({'text': 'review', 'label': 'sentiment'})
ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

# get model response
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": ilql_config["forward_batch_size"]
}

sentiment_pipe = pipeline("sentiment-analysis","samibg/distilbert-imdb", device=pipe_device)
# get models

# TODO: Rename and fix models
gpt2_model = GPT2HeadWithQValueModel.from_pretrained('gpt2')
gpt2_pi_beta = GPT2HeadWithQValueModel.from_pretrained('gpt2')

gpt2_tokenizer = AutoTokenizer.from_pretrained(ilql_config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

wandb.watch(gpt2_model, log='all')
gpt2_model.to(device)
gpt2_pi_beta.to(device)

class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
    def __call__(self):
        return np.random.choice(self.values)
    
input_size = LengthSampler(ilql_config["txt_in_min_len"], ilql_config["txt_in_max_len"])
output_size = LengthSampler(ilql_config["txt_out_min_len"], ilql_config["txt_out_max_len"])

def tokenize(sample):
    sample["tokens"] = gpt2_tokenizer.encode(sample["review"])[:input_size()]
    sample["query"] = gpt2_tokenizer.decode(sample["tokens"])
    return sample

ds = ds.map(tokenize, batched=False)

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataloader = torch.utils.data.DataLoader(ds, batch_size=ilql_config['batch_size'], collate_fn=collater)
# Training loop

# initialize trainer
ilql_trainer = ILQLTrainer(gpt2_model, gpt2_tokenizer, **ilql_config)

total_ilql_epochs = int(np.ceil(ilql_config["steps"]/ilql_config['batch_size']))

for epoch, batch in tqdm(zip(range(total_ilql_epochs), iter(dataloader))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    
    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(total_ilql_epochs['batch_size']):
        gen_len = output_size()
        response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),
                                       max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time()-t

    #### Compute sentiment score
    t = time.time()
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = torch.tensor([output[1]["score"] for output in pipe_outputs]).to(device)
    timing['time/get_sentiment_preds'] = time.time()-t
    
    #### Run ILQL step 
    t = time.time()
    stats = ilql_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
    logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)