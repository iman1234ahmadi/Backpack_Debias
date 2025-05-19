import os
import ir_datasets
import pandas as pd
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from concurrent.futures import as_completed
import argparse
import torch
import sys
import argparse

# 1) Build your parser
parser = argparse.ArgumentParser(description="A simple ID‐driven script")

# 2) Declare your flags
parser.add_argument(
    "--id",
    type=int,
    default=-1,
    help="An integer identifier (default: -1)"
)

parser.add_argument(
    "--precent",
    type=float,
    default=1.0,
    help="A float identifier (default: 0.1)"
)
parser.add_argument(
    "--GPU_CUDA",
    type=int,
    default=0,
    help="An integer identifier (default: 0)"
)

parser.add_argument(
    "--model_name",
    type=str,
    default="BP_ranker_freeze_sensenet_final",
    help="A string identifier (default: BP_ranker_freeze_sensenet_final)"
)

args = parser.parse_args()
run_id= args.id
precent = args.precent
model_name = args.model_name
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_CUDA)

if run_id == -1:
    print("No ID provided. Please provide a valid ID.")
    sys.exit(1)

x_values = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
alpha = x_values[run_id]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(f'/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/Dataset/Final_Test2/{model_name}/alpha{alpha}', exist_ok=True)
sys.stdout = open(f'/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/Dataset/Final_Test2/{model_name}/alpha{alpha}/out.out', 'w')
sys.stderr = open(f'/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/Dataset/Final_Test2/{model_name}/alpha{alpha}/logs.rr', 'w')

print(f"Device: {device}")
print(f"Alpha: {alpha}")

dataset = ir_datasets.load("msmarco-passage/dev")
queries = dataset.queries_iter()
docs = dataset.docs_iter()
qrels = dataset.qrels_iter()

path = "/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/Dataset/Test-Datasets/GenderBias_IR/Base/GenderBias_IR/sample_trec_runs/msmarco_passage/bm25.run"

# Define column names for clarity
column_names = ['query_id', 'static', 'doc_id', 'rank', 'score', 'system']

# Read the data into a pandas DataFrame
df = pd.read_csv(path, sep='\s+', names=column_names, usecols=['query_id', 'doc_id'])

# Group the document IDs by query ID
query_doc_mapping = df.groupby('query_id')['doc_id'].apply(list).to_dict()

query_doc_mapping = df.groupby('query_id')['doc_id'].apply(list).reset_index()


sys.path.append("/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/")
import json
from org_backpack.modeling_backpack_gpt2 import BackpackGPT2LMHeadModel
from org_backpack.configuration_backpack_gpt2 import BackpackGPT2Config
import torch.nn as nn
from transformers import AutoTokenizer
import torch
from debiaser_vector import DebiaserVector
from collections import Counter
import numpy as np

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

# Load JSON file
with open('/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/org_backpack/config.json', 'r') as file:
    conf = json.load(file)

config = BackpackGPT2Config(**conf) 
model = BackpackGPT2LMHeadModel(config)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

checkpoint_path = '/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/org_backpack/pytorch_model.bin'
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

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

# BP_model_with_sigmoid.load_state_dict(torch.load('/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/rankers/BP_ranker_9Jun/BP_model.pth',weights_only=True))
# BP_model_with_sigmoid.eval()

BP_model_with_sigmoid.load_state_dict(
    torch.load(
        f'/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/models/rankers/{model_name}/BP_model.pth',
        map_location=device,
        weights_only=True
    )
)

BP_model_with_sigmoid.to(device)
BP_model_with_sigmoid.eval()


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
# print(f"Probabilities: {probabilities}")


# Count non-zero values
non_zero_count = np.count_nonzero(probabilities)

# print(f"Non-zero entries: {non_zero_count}")
# Create an all-ones vector
ones_vector = np.ones(len(all_indices))
bias_param_vecor =  np.ones(len(all_indices))

## Case 1
bias_param_vecor[max_indices[0]] = bias_param_vecor[max_indices[0]]*alpha
bias_param_vecor[max_indices[1]] = bias_param_vecor[max_indices[1]]*alpha
# bias_param_vecor[max_indices[2]] = bias_param_vecor[max_indices[2]]*alpha

dataset = ir_datasets.load("msmarco-passage/dev")
queries = dataset.queries_iter()
docs = dataset.docs_iter()

queries_pd = pd.DataFrame(dataset.queries_iter())
doc_pd = pd.DataFrame(dataset.docs_iter())

# Ensure 'doc_id' is of the correct type (int) and there are no leading/trailing spaces
doc_pd['doc_id'] = doc_pd['doc_id'].astype(int)


def get_model(model, bias_param_vecor, device):
    model.edit_params(bias_param_vecor)
    print(f"Model loaded on {device}")
    model.to(device)
    print(model.backpack.parameters_of_contextualization)
    return model

def score_documents(model, input_ids):
    # Forward pass on GPU or CPU
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits.squeeze().cpu().numpy()
    score = logits.mean()

    return score 

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def ranker_on_BPModel_parallel(model, query_id, related_docs, tokenizer, device, queries_pd, doc_pd):
    model.eval()
    query = queries_pd.loc[queries_pd['query_id'] == query_id]['text'].values[0]
    
    future_to_doc = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        for doc_id in related_docs:
            try:
                document = doc_pd.loc[doc_pd['doc_id'] == doc_id]['text'].values[0]
                input_text = query + " <EOS> " + document
                input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            except Exception as e:
                print(e)
                print(query_id, doc_id)
                future = executor.submit(lambda: 0.0)
                future_to_doc[future] = doc_id
                continue
            future = executor.submit(score_documents, model, input_ids)
            future_to_doc[future] = doc_id

        # Convert the generator to a list and wrap with tqdm.
        futures = list(as_completed(future_to_doc))
        scores = []
        for i, future in enumerate(futures):
            # print(f"Processing {i+1}/{len(futures)}: Ranking Documents for Query {query_id}", end="\r")
            doc_id = future_to_doc[future]
            try:
                score = future.result()
            except Exception as exc:
                print(f"\nDoc {doc_id} generated an exception: {exc}")
                score = 0.0
            scores.append((doc_id, score))
        # print() 
    
    # Sort the documents by score in descending order.
    doc_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return doc_scores

def test_on_BPModel(BP_model_with_sigmoid, query_doc_mapping, bias_param_vecor, device, queries_pd, doc_pd):
    output = []
    local_model = get_model(BP_model_with_sigmoid, bias_param_vecor, device)
    for i, (_, row) in enumerate(tqdm(query_doc_mapping.iterrows(), total=len(query_doc_mapping), desc="Processing Queries")):
        sorted_docs = ranker_on_BPModel_parallel(local_model, row['query_id'], row['doc_id'], tokenizer, device, queries_pd, doc_pd)
        output.extend(
            [f"{row['query_id']} Q0 {doc_id} {rank} {score} neural_model"
             for rank, (doc_id, score) in enumerate(sorted_docs, start=1)]
        )
    return output

# Cast queries_pd['query_id'] to int
queries_pd['query_id'] = queries_pd['query_id'].astype(int)
# Shuffle query_doc_mapping
query_doc_mapping = query_doc_mapping.sample(frac=1).reset_index(drop=True)

# Use "precent" of the data for testing
test_size = int(precent * len(query_doc_mapping))
query_doc_mapping = query_doc_mapping[:test_size]
# Print the number of queries being processed
print(f"Number of queries being processed: {len(query_doc_mapping)}")

output_bias = test_on_BPModel(BP_model_with_sigmoid, query_doc_mapping, bias_param_vecor, device, queries_pd, doc_pd)

# Specify the file path (this will save it as a .run file)
file_path = f'/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/Dataset/Final_Test2/{model_name}/alpha{alpha}/{model_name}_alpha{alpha}.run'

# Open the file in write mode and save the content
with open(file_path, 'w') as f:
    for line in output_bias:
        f.write(line + '\n')  # Add newline after each string

print(f"Output saved to {file_path}")

# Close the files at the end of the script
sys.stdout.close()
sys.stderr.close()