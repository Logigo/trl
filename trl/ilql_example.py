# imports
import torch
import numpy as np
from trl.gpt2_ilql import GPT2HeadWithQValueModel, respond_to_batch
from trl.ilql import ILQLTrainer
import wandb
from transformers import AutoTokenizer, pipeline
from trl.core import build_bert_batch_from_txt, listify_batch
from datasets import load_dataset
import time
from tqdm import tqdm

# NOTE: Straight up copied from https://github.com/lvwerra/trl/blob/master/nbs/04-gpt2-sentiment-ppo-training.ipynb
ilql_config = {
    'steps': 300, 'batch_size': 3, 'forward_batch_size': 1, 'model_name': 'lvwerra/gpt2-imdb', 
    "txt_in_min_len": 2, "txt_in_max_len": 8, "txt_out_min_len": 4, "txt_out_max_len": 16
}

SAVE_FREQUENCY = 10
SAVE_PATH = './'
TOTAL_VALIDATION_RUNS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

wandb.init(name='training_run1', project='ilql-gpt2', config=ilql_config, )


# get model response
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": ilql_config["forward_batch_size"]
}

sentiment_pipe = pipeline("sentiment-analysis","lvwerra/distilbert-imdb", device=pipe_device)
# get models
# TODO: Ask charlie about 
#   Target Q Heads

# TODO: Find a way to pass PerUtterance OR PerToken to GPT2 Model
ilql_model = GPT2HeadWithQValueModel.from_pretrained('gpt2')
gpt2_pi_beta = GPT2HeadWithQValueModel.from_pretrained('gpt2')

gpt2_tokenizer = AutoTokenizer.from_pretrained(ilql_config['model_name'])
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

wandb.watch(ilql_model, log='all')
ilql_model.to(device)
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

# load imdb with datasets
ds_train, ds_val, ds_test = load_dataset('imdb', split=['train[:8%]', 'train[8%:9%]', 'test[-1%:]'])


def process_data(dataset):
    dataset = dataset.rename_columns({'text': 'review', 'label': 'sentiment'})
    dataset = dataset.filter(lambda x: len(x["review"])>200, batched=False)
    dataset = dataset.map(tokenize, batched=False)
    return dataset

ds_train = process_data(ds_train)
ds_val = process_data(ds_val)
ds_test = process_data(ds_test)

gen_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id
}

def collater(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=ilql_config['batch_size'], collate_fn=collater)
dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=ilql_config['batch_size'], collate_fn=collater)
dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=ilql_config['batch_size'], collate_fn=collater)

# Training loop

# initialize trainer
ilql_trainer = ILQLTrainer(ilql_model, gpt2_tokenizer, **ilql_config)
total_ilql_epochs = int(np.ceil(ilql_config["steps"]/ilql_config['batch_size']))

for epoch, batch in tqdm(zip(range(total_ilql_epochs), iter(dataloader_train))):
    logs, timing = dict(), dict()
    t0 = time.time()
    query_token_tensors = [torch.tensor(t).long().to(device) for t in batch["tokens"]]
    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for i in range(ilql_config['batch_size']):
        gen_len = output_size()
        with torch.no_grad():
            response = ilql_model.generate(query_token_tensors[i].unsqueeze(dim=0),
                                        max_new_tokens=gen_len, **gen_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [gpt2_tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time()-t

    #### Compute sentiment score
    t = time.time()
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = torch.tensor([output["score"] if output['label'] == 'POSITIVE' else 1. - output["score"] for output in pipe_outputs]).to(device)
    timing['time/get_sentiment_preds'] = time.time()-t
    
    #### Run ILQL step 
    t = time.time()
    stats = ilql_trainer.step(query_token_tensors, response_tensors, rewards)
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

    if epoch % (total_ilql_epochs // TOTAL_VALIDATION_RUNS) == 0:
        validation_stats = []
        # For every batch, run inference on every input. Record response tokens for each input.
        # Find sentiment for every item in batch and record
        for v_batch in tqdm(iter(dataloader_val)):
            query_token_tensors = [torch.tensor(t).long().to(device) for t in v_batch["tokens"]]
            # Run inference on batch and update validation batch
            inference_args = [gpt2_pi_beta, ilql_model, ilql_config['txt_out_max_len'], 8]
            inference_token_response = [respond_to_batch(*inference_args, q) for q in query_token_tensors]
            inference_decoded_response = [gpt2_tokenizer.decode(r.squeeze()) for r in inference_token_response]
            v_batch['response'] = inference_decoded_response
            # Aggregate ILQL inference output and get sentiment results
            texts = [q + r for q,r in zip(v_batch['query'], v_batch['response'])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            v_batch_rewards = torch.tensor([output["score"] if output['label'] == 'POSITIVE' else 1. - output["score"] for output in pipe_outputs]).to(device)
            validation_stats.append({
                'queries': v_batch['query'], 'responses': v_batch['response'],
                'query_tokens': query_token_tensors, 'responses_tokens': inference_token_response,
                'rewards': v_batch_rewards
            })
        # Log results
        v_rewards = [] ; [v_rewards.extend(v['rewards']) for v in validation_stats]
        logs['validation_rewards_mean'] = torch.mean(v_rewards).cpu().numpy()
        logs['validation_rewards_std'] = torch.std(v_rewards).cpu().numpy()
        logs['validation_rewards'] = v_rewards # TODO. Graph pareto frontier?  x - nll for generation. y - beta
    
    wandb.log(logs)

# Evaluating performance on test dataset - Run inference with ILQL model, and pass the results through bert to see 
#  +ivity rate. 