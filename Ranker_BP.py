import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"
import torch 
import sys 
import wandb
import os 
import random
import torch
import torch.nn as nn
from datasets import load_dataset 
import numpy as np
import gzip
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from models.org_backpack.modeling_backpack_gpt2_training import BackpackGPT2LMHeadModel
from models.org_backpack.configuration_backpack_gpt2 import BackpackGPT2Config
import json
from accelerate import notebook_launcher


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freeze_sensenet = True 
    model_name = 'BP_ranker_freeze_sensenet_final'
    batch_size = 10
    # Load the MS MARCO dataset from Hugging Face
    dataset = load_dataset("microsoft/ms_marco", "v2.1", 
                        cache_dir='/mnt/raid10/ak-research-01/ak-research-01/codes/Backpack-LM/cache')

    def prepare_data(dataset, split, m=36):
        """
        Prepare data for the specified split (train, validation, test).

        Args:
            dataset (dict): The dataset containing the splits.
            split (str): The split to prepare ('train', 'validation', or 'test').
            m (int): The number of passages to include (1 positive + m-1 negatives).

        Returns:
            list: A list of tuples (query, positive_text, negative_texts).
        """
        def filter_data(data):
            """
            Filter data to ensure each entry has exactly one positive and nine negative answers.
            Prints a message for entries that do not meet the requirement.
            """
            filtered_data = []
            for entry in data:
                if len(entry[2]) == 9:
                    filtered_data.append(entry)
                else:
                    pass
            return filtered_data

        data = []
        if split not in dataset:
            raise ValueError(f"Split '{split}' not found in the dataset.")

        for example in dataset[split]:
            query = example['query']
            if isinstance(example['passages'], dict):
                is_selected_list = example['passages'].get('is_selected', [])
                passage_text_list = example['passages'].get('passage_text', [])
                if is_selected_list and passage_text_list:
                    pos_indices = [i for i, val in enumerate(is_selected_list) if val == 1]
                    if pos_indices:
                        pos_index = pos_indices[0]
                        pos_text = passage_text_list[pos_index]
                    else:
                        continue
                else:
                    continue
            else:
                continue

            # Select m-1 negative documents
            neg_indices = [i for i, val in enumerate(is_selected_list) if val == 0]
            neg_texts = [passage_text_list[i] for i in random.sample(neg_indices, min(len(neg_indices), m - 1))]
            data.append((query, pos_text, neg_texts))

        data = filter_data(data)
        return data

    class RankingDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            query, pos_text, neg_texts = self.data[idx]

            # Combine positive and negative texts into a list
            texts = [pos_text] + neg_texts

            # Tokenize the query with all texts
            encoding = self.tokenizer(
                [query] * len(texts),
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].squeeze(0)  # (num_docs, seq_len)
            attention_mask = encoding['attention_mask'].squeeze(0)  # (num_docs, seq_len)


            # Create labels tensor
            labels = torch.zeros(len(texts))
            labels[0] = 1  # Positive label for the first text 

            return {
                'input_ids': input_ids,  # (num_docs, seq_len)
                'attention_mask': attention_mask,  # (num_docs, seq_len)
                'labels': labels  # (num_docs,)
            }

    class ListwiseSoftmaxCrossEntropyLoss(nn.Module):
        def __init__(self):
            super(ListwiseSoftmaxCrossEntropyLoss, self).__init__()

        def forward(self, logits, labels):
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = -torch.mean(torch.sum(labels * log_probs, dim=1))

            return loss 

    def compute_metrics(predictions, labels):
        def mrr_at_k(predictions, labels, k=10):
            ranks = np.argsort(-predictions, axis=1)
            rr = 0.0
            for i in range(len(labels)):
                if 1 in labels[i][ranks[i][:k]]:
                    rr += 1 / (np.where(labels[i][ranks[i][:k]] == 1)[0][0] + 1)
            return rr / len(labels)   

        def ndcg_at_k(predictions, labels, k=10):
            def dcg(scores):
                return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

            ndcg = 0.0
            for i in range(len(labels)):
                sorted_preds = np.argsort(-predictions[i])[:k]
                ideal_sorted_labels = np.sort(labels[i])[::-1][:k]
                actual_dcg = dcg(labels[i][sorted_preds])
                ideal_dcg = dcg(ideal_sorted_labels)
                ndcg += actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
            return ndcg / len(labels)

        def average_precision_at_k(predictions, labels, k=10):
            ap = 0.0
            for i in range(len(labels)):
                sorted_preds = np.argsort(-predictions[i])[:k]
                rel_scores = labels[i][sorted_preds]
                if np.sum(rel_scores) == 0:
                    continue
                precision_at_i = [np.sum(rel_scores[:i+1]) / (i+1) for i in range(len(rel_scores))]
                ap += np.mean(precision_at_i) * (np.sum(rel_scores) / np.sum(labels[i]))
            return ap / len(labels)

        mrr = mrr_at_k(predictions, labels)
        ndcg5 = ndcg_at_k(predictions, labels, k=5)
        ndcg10 = ndcg_at_k(predictions, labels, k=10)
        map_score = average_precision_at_k(predictions, labels, k=10)

        return {
            "mrr@10": mrr,
            "ndcg@5": ndcg5,
            "ndcg@10": ndcg10,
            "map@10": map_score
        }

    class BackPackWithSigmoid(BackpackGPT2LMHeadModel):
        def __init__(self, bp_model, freeze_sensenet=False):
            super(BackpackGPT2LMHeadModel, self).__init__(bp_model.config)
            self.backpack = bp_model.backpack  # Copy the transformer part of the model
            self.lm_head = bp_model.lm_head  # Copy the language modeling head
            self.sigmoid = nn.Sigmoid()  # Sigmoid activation

            # Freeze the parameters of self.backpack.sense_network
            if freeze_sensenet:
                for param in self.backpack.sense_network.parameters():
                    param.requires_grad = False

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            # Get the outputs from the base GPT-2 model
            outputs = super().forward(input_ids=input_ids)#, attention_mask=attention_mask, **kwargs)
            
            # Apply sigmoid activation to the logits
            logits = outputs.logits
            logits = self.sigmoid(logits)
            outputs.logits = logits
            return outputs


    # Load JSON file
    with open('/mnt/models/org_backpack/config.json', 'r') as file:
        conf = json.load(file)

    config = BackpackGPT2Config(**conf) 
    model = BackpackGPT2LMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    checkpoint_path = '/mnt/models/org_backpack/pytorch_model.bin'
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu')) 
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()} 
    model.load_state_dict(state_dict=filtered_state_dict)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add special tokens
    special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>'}
    tokenizer.add_special_tokens(special_tokens_dict)

    model.lm_head = nn.Linear(in_features=768, out_features=1, bias=False)  

    # Wrap it with the sigmoid
    model_with_sigmoid = BackPackWithSigmoid(model, freeze_sensenet=freeze_sensenet).to(device) 

    # Prepare data
    train_data = prepare_data(dataset, "train", m=36)
    val_data = prepare_data(dataset, "validation", m=36)
    # test_data = prepare_data(dataset, "test", m=36)

    # Create datasets
    train_dataset = RankingDataset(train_data, tokenizer)
    val_dataset = RankingDataset(val_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Replace the default loss with Listwise Softmax Cross Entropy
    LSC_loss = ListwiseSoftmaxCrossEntropyLoss()

    class CustomTrainer(Trainer):
        def __init__(self, model, args, train_dataloader, eval_dataloader, **kwargs):
            super().__init__(model, args, **kwargs)
            self.train_dataloader = train_dataloader
            self.eval_dataloader = eval_dataloader

        def get_train_dataloader(self):
            return self.train_dataloader

        def get_eval_dataloader(self, eval_dataset=None):
            return self.eval_dataloader

        def compute_loss(self, model, inputs, num_items_in_batch=0, return_outputs=False):
            # Forward pass
            torch.cuda.empty_cache()

            input_ids = inputs['input_ids']  # (batch_size, num_docs, seq_len)
            attention_mask = inputs['attention_mask']  # (batch_size, num_docs, seq_len)
            labels = inputs['labels']  # (batch_size, num_docs)

            # Flatten the inputs for the model
            batch_size, num_docs, seq_len = input_ids.size() 
            input_ids = input_ids.view(-1, seq_len)  # (batch_size * num_docs, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)  # (batch_size * num_docs, seq_len)

            # Pass through the model
            outputs = model(input_ids)

            logits = outputs.logits[:,-1,:].view(batch_size,num_docs)  # (batch_size, num_docs)

            if logits.shape != labels.shape:
                raise ValueError(f"Logits shape {logits.shape} does not match labels shape {labels.shape}")

            # Compute custom loss
            loss = LSC_loss(logits, labels)

            return (loss, outputs) if return_outputs else loss


    wandb.init(project="BackPack",name=model_name) 


    # Assuming model_with_sigmoid is defined elsewhere
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make the model parallel
    # model_with_sigmoid = nn.DataParallel(model_with_sigmoid)
    model_with_sigmoid.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'/mnt/models/rankers/{model_name}',
        num_train_epochs=4,
        # max_steps=10,
        per_device_train_batch_size=batch_size,  # Reduce batch size
        per_device_eval_batch_size=batch_size,   # Reduce batch size
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir=f'/mnt/models/rankers/{model_name}/logs',
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500000,  # Evaluate every 100 steps
        learning_rate=5e-6,
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 batches
        fp16=True,  # Enable mixed precision training
        report_to='wandb',
        seed=42,
        save_safetensors=False,
        save_steps=100000)

    trainer = CustomTrainer(
        model=model_with_sigmoid,
        args=training_args,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        eval_dataset=val_dataset,
        compute_metrics=None,
    )

    # # Start training
    trainer.train()

    # trainer.save_model('/mnt/raid10/ak-research-01/ak-research-01/RPO/cache0/BP/Backpack-LM-debias/models/rankers')
    torch.save(model_with_sigmoid.state_dict(), f'/mnt/rankers/{model_name}/BP_model.pth')
    print('model saved!')

    sys.stdout.close()
    sys.stderr.close()


if __name__ == "__main__":
    notebook_launcher(main, args=(), num_processes=torch.cuda.device_count()) 
