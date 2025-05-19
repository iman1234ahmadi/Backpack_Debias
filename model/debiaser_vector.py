import pandas as pd
import re
import torch
import torch.nn.functional as F

class DebiaserVector:
    def __init__(self, model, device, tokenizer, data_path, name="crows_pair"): 
        self.model = model
        self.device = device
        self.name = name
        self.tokenizer = tokenizer

    def load_data(self, data_path):
        if not data_path:
            raise ValueError("Data path is empty")
        return pd.read_csv(data_path, index_col=0)
    
    def process_data(self, df, bias_type= "gender",):
        if self.name == "crows_pair":
            return self.process_crows_pair(df, bias_type)
        else:
            raise ValueError("New dataset not supported")
        
    
    def process_crows_pair(self, df, bias_type):
        df_biased = df[df['bias_type'] == bias_type]

        filtered_rows = []

        # Iterate through the DataFrame
        for index, row in df_biased.iterrows():
            sent_more_words = row['sent_more'].split()
            sent_less_words = row['sent_less'].split()

            # Ensure both sentences have the same length
            if len(sent_more_words) == len(sent_less_words):
                # Find the word that differs
                differing_indices = [i for i in range(len(sent_more_words)) if sent_more_words[i] != sent_less_words[i]]

                # If there's exactly one differing word, add this row to the new DataFrame
                if len(differing_indices) == 1:
                    differing_index = differing_indices[0]
                    filtered_rows.append({
                        'sent_more': row['sent_more'],
                        'sent_less': row['sent_less'],
                        'differing_index': differing_index
                    })

        # Create a new DataFrame with the filtered rows
        df_filtered = pd.DataFrame(filtered_rows)
        
        return df_filtered

    def get_bias_topics(self, df_filtered):
        bias_topic = []
        bias_topic_txt = []
        for i in range(df_filtered.shape[0]):

            bias_topic.append(
                (df_filtered.iloc[i].sent_more.split()[df_filtered.iloc[i].differing_index],
                df_filtered.iloc[i].sent_less.split()[df_filtered.iloc[i].differing_index])
            )
            
            bias_topic_txt.append(
                f"{df_filtered.iloc[i].sent_more.split()[df_filtered.iloc[i].differing_index]} {df_filtered.iloc[i].sent_less.split()[df_filtered.iloc[i].differing_index]}"
            )

            # Normalize and deduplicate
        normalized_data = set(normalize_string(item) for item in bias_topic_txt)

        # Convert back to list if needed
        unique_data = list(normalized_data)
        return unique_data

    
    def get_bias_indices(self, unique_data):
        # A list to store the indices of the bias_topic-related senses
        bias_topic_sense_indices = []

        for i in range(len(unique_data)):
            # Step 1: Tokenize and encode the input text
            input_text = unique_data[i]
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

            # Step 2: Pass input_ids through the model to get the sense vectors
            with torch.no_grad():
                torch_out = self.model(input_ids, position_ids=None)
            sense_vectors = torch_out.senses.squeeze(0)  # torch_out should be the tensor containing the 
            sense_vectors_normalized = F.normalize(sense_vectors, p=2, dim=-1)
            cosine_similarities = torch.bmm(
                sense_vectors_normalized[:, 0, :].unsqueeze(1), 
                sense_vectors_normalized[:, 1, :].unsqueeze(2)
            ).squeeze()  

            # Find the maximum cosine similarity, which will be along the diagonal in this case
            max_cosine_sim, max_diff_index = torch.min(cosine_similarities, dim=0)


            # Store the index of the sense that shows the largest difference
            bias_topic_sense_indices.append(max_diff_index.item())

        return bias_topic_sense_indices





def normalize_string(s):
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    return s