import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import json
import gzip
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.org_backpack.modeling_backpack_gpt2_training import BackpackGPT2LMHeadModel
from models.org_backpack.configuration_backpack_gpt2 import BackpackGPT2Config

from collections import Counter
import argparse
import jsonlines
from collections import defaultdict
import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import pandas as pd
import logging
import pathlib, os
import glob
from tqdm import tqdm
from huggingface_hub import login

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BackPackWithSigmoid(BackpackGPT2LMHeadModel):
    def __init__(self, bp_model):
        super(BackpackGPT2LMHeadModel, self).__init__(bp_model.config)
        self.backpack = bp_model.backpack  # Copy the transformer part of the model
        self.lm_head = bp_model.lm_head  # Copy the language modeling head
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Get the outputs from the base GPT-2 model
        outputs = super().forward(input_ids=input_ids)#, attention_mask=attention_mask, **kwargs)
        
        # Apply sigmoid activation to the logits
        logits = outputs.logits
        logits = self.sigmoid(logits)
        
        # Replace the logits in the output with the sigmoid-applied logits
        outputs.logits = logits
        
        return outputs


with open('/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/org_backpack/config.json', 'r') as file:
    conf = json.load(file)

config = BackpackGPT2Config(**conf) 
model = BackpackGPT2LMHeadModel(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

checkpoint_path = '/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/org_backpack/pytorch_model.bin'
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Filter the external state_dict to retain only the keys present in the model's state_dict
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()} 

model.load_state_dict(state_dict=filtered_state_dict)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add special tokens
special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>'}
tokenizer.add_special_tokens(special_tokens_dict)

model.lm_head = nn.Linear(in_features=768, out_features=1, bias=False)

# Wrap it with the sigmoid
BP_model_with_sigmoid = BackPackWithSigmoid(model)

BP_model_with_sigmoid.load_state_dict(torch.load('/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/rankers/BP_ranker_freeze_sensenet_final/BP_model.pth'))
BP_model_with_sigmoid.eval()
from debiaser_vector_copy import DebiaserVector

data_path = '/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/Dataset/crows_pairs/data/crows_pairs_anonymized.csv'
bias_type = "gender"
debiaser = DebiaserVector(model=BP_model_with_sigmoid.to('cuda'), tokenizer=tokenizer, device='cuda',data_path = data_path)
data_frame = debiaser.load_data(data_path)

data_frame = debiaser.process_data(data_frame)
unique_data = debiaser.get_bias_topics(data_frame)
bias_topic_sense_indices = debiaser.get_bias_indices(unique_data)

# Count frequencies of each index
counter = Counter(bias_topic_sense_indices)

# Ensure all indices from 0 to 15 are included
all_indices = list(range(16))
total_count = len(bias_topic_sense_indices)  # Total number of samples
frequencies = [counter.get(i, 0) for i in all_indices]

# Convert frequencies to probabilities
probabilities = [freq / total_count for freq in frequencies]

probabilities_array = np.array(probabilities)

# Get the indices of the three maximum values
max_indices = np.argsort(probabilities_array)[-3:][::-1]

# print(max_indices)
print(f"Probabilities: {probabilities}")


# Count non-zero values
non_zero_count = np.count_nonzero(probabilities)

print(f"Non-zero entries: {non_zero_count}")
# Create an all-ones vector
ones_vector = np.ones(len(all_indices))
bias_param_vecor =  np.ones(len(all_indices))

# change this for different weights
constants = [0,0.1, 0.3, 0.5, 0.7,0.9,1.0]
bias_param_vecor[max_indices[0]] = bias_param_vecor[max_indices[0]]*constants[0]
bias_param_vecor[max_indices[1]] = bias_param_vecor[max_indices[1]]*constants[0]
bias_param_vecor[max_indices[2]] = bias_param_vecor[max_indices[2]]*constants[0]

bias_param_vecor.tolist()

def get_model(model, bias_param_vecor, device):
    model.edit_params(bias_param_vecor)
    print(f"Model loaded on {device}")
    model.to(device)
    print(model.backpack.parameters_of_contextualization)
    return model

model_bias = get_model(BP_model_with_sigmoid,bias_param_vecor,device)


def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")

def read_jsonl_file(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def cleanup_id(id_text):
    if type(id_text) == int:
        return str(id_text)
    elif '_' in id_text:
        return id_text.split('_')[-1]
    else:
        raise NotImplementedError

def convert_to_result_format(data):
    output = {}
    for instance in data: # to do: inspect more
        qid = str(instance['qid'])
        input_q = instance['q_text']
        psg_ids = [str(x['pid']) for x in instance['bm25_results']]
        scores = [x['bm25_score'] for x in instance['bm25_results']]
        output[qid] = {}
        for psgid, score in zip(psg_ids, scores):
            output[qid][psgid] = float(score)
    return output

def check_dup(data, key='q_text'):
    new = []
    qs = set()
    for d in data:
        if d['q_text'] not in qs:
            new.append(d)
            qs.add(d['q_text'])
    if len(data) != len(new):
        print(f"Original data len {len(data)}, removed dup to {len(new)}")
    return new
    #return new
def check_100(data):
    new = []
    for instance in data:
        if len(instance['bm25_results']) > 100:
            instance['bm25_results'] = instance['bm25_results'][:100]
        new.append(instance)
    return new

def setup():

    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def format_res_for_print(acc, ndcg, _map, recall, pre, mrr):
    out = ''
    out+= f"\nAccuracy@1/5/10/20/50/100: {acc['Accuracy@1']}, {acc['Accuracy@5']}, {acc['Accuracy@10']}, {acc['Accuracy@20']}, {acc['Accuracy@50']}, {acc['Accuracy@100']}, "
    out+= f"\nNDCG@1/5/10/20/50/100: {ndcg['NDCG@1']}, {ndcg['NDCG@5']}, {ndcg['NDCG@10']}, {ndcg['NDCG@20']}, {ndcg['NDCG@50']}, {ndcg['NDCG@100']}, "
    out+= f"\nMRR@1/5/10/20/50/100: {mrr['MRR@1']}, {mrr['MRR@5']}, {mrr['MRR@10']}, {mrr['MRR@20']}, {mrr['MRR@50']}, {mrr['MRR@100']}, "
    out+= f"\nRECALL@1/5/10/20/50/100: {recall['Recall@1']}, {recall['Recall@5']}, {recall['Recall@10']}, {recall['Recall@20']}, {recall['Recall@50']}, {recall['Recall@100']}, "
    out+= f"\nPrecision@1/5/10/20/50/100: {pre['P@1']}, {pre['P@5']}, {pre['P@10']}, {pre['P@20']}, {pre['P@50']}, {pre['P@100']}, "
    out+= f"\nMAP@1/5/10/20/50/100: {_map['MAP@1']}, {_map['MAP@5']}, {_map['MAP@10']}, {_map['MAP@20']}, {_map['MAP@50']}, {_map['MAP@100']}, \n"
    return out

def make_dummy_results(corpus, queries):
    query_keys = list(queries.keys())
    corpus_keys = list(corpus.keys())
    out = {}
    for key in query_keys:
        out[key] = {ck: 0.5 for ck in corpus_keys}
    return out

def remove_nan(results):
    new_res = {}
    for query_key in results.keys():
        out = {}
        for i, corpus_key in enumerate(results[query_key].keys()):
            out[corpus_key] = 100 - i
        new_res[query_key] = out
    return new_res

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def do_evaluation(queries, qrels, corpus, results=None, mode='ours'):
    k_values = [1, 5, 10, 20, 50, 100]
    retriever = EvaluateRetrieval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reranked_results = {}
    query_count = 0
    for query_id in tqdm(results, desc="Processing queries"):
        query_text = queries[query_id]
        doc_scores = results[query_id]

        # Prepare inputs for BackPackWithSigmoid
        input_ids_list = []
        attention_mask_list = []
        doc_ids = list(doc_scores.keys())

        for doc_id in doc_ids:
            doc_text = corpus[doc_id]['text']
            # Tokenize query and document (you need to implement or use a tokenizer here)
            inputs = tokenizer(query_text, doc_text,padding='max_length',truncation=True,
            max_length=512,return_tensors='pt')
            input_ids_list.append(inputs['input_ids'])
            attention_mask_list.append(inputs['attention_mask'])

        # Stack inputs
        input_ids = torch.cat(input_ids_list, dim=0).to(BP_model_with_sigmoid.device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(BP_model_with_sigmoid.device)

        # Forward pass through BackPackWithSigmoid
        with torch.no_grad():
            outputs = model_bias(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].squeeze()  # Adjust based on your model's output

        # Update scores in results
        reranked_results[query_id] = {doc_id: float(logits[i]) for i, doc_id in enumerate(doc_ids)}
        query_count += 1
    
    # Evaluate reranked results
    ndcg, _map, recall, precision = retriever.evaluate(qrels, reranked_results, k_values)
    mrr = retriever.evaluate_custom(qrels, reranked_results, k_values, metric='mrr')
    hits = retriever.evaluate_custom(qrels, reranked_results, k_values, metric='top_k_accuracy')

    ndcg_10 = ndcg['NDCG@10']
    _map_10 = _map['MAP@10']
    mrr_10 = mrr['MRR@10']
    out_string = format_res_for_print(hits, ndcg, _map, recall, precision, mrr)
    return ndcg_10,mrr_10,_map_10, out_string
    
def make_corpus(data):
    res = {}
    for line in data:
        ctxs = line['bm25_results']
        for subline in ctxs:
            res[str(subline['pid'])] = {'text': subline['text'], 'title': 'none'}
    return res

def run_rerank_eval(data_path, mode='ours', combined=False):
    if combined:
        data = data_path
    else:
        data = read_jsonl_file(data_path)
    data = check_dup(data)
    data = check_100(data)
    results = convert_to_result_format(data)
    corpus = make_corpus(data)
    queries = {}
    qrels = {}
    ## making queries and qrels
    for line in tqdm(data, desc="Preparing queries and qrels"):
        id_text = str(line['qid'])
        queries[id_text] = line['q_text']
        qrels[id_text] = line['qrels']
    ndcg_10,mrr_10,_map_10, out_string = do_evaluation(queries, qrels, corpus, results=results, mode=mode)
    if not combined:
        print(f"For {data_path}")
    print(f"Evaluation results :")
    print(out_string)
    print(f"NDCG@10: ")
    print(ndcg_10)
    print(f"mrr@10: ")
    print(mrr_10)
    print(f"MAP@10: ")
    print(_map_10)
    return ndcg_10,mrr_10,_map_10, out_string



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='msmarco.jsonl')
    parser.add_argument('--mode', default='ours', type=str)
    parser.add_argument('--combine', action='store_true')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args, _ = parser.parse_known_args() 
    
    setup()
    if args.combine:
        paths = glob.glob(f"{args.path}/*.jsonl")
        print("BP_ranker_no_freeze_sensenet_final")
        print(f'Combine = True, paths: {paths}')
        full_data = []
        for path in paths:
            data = read_jsonl_file(path)
            full_data += data
        print(f"Combined full len: {len(full_data)}")
        run_rerank_eval(full_data, mode=args.mode, combined=True)
    else:
        run_rerank_eval(args.path, mode=args.mode, combined=False) 