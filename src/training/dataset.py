import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ToxicDataset(Dataset):
    
    def __init__(self, file_paths: list[str] | str, tokenizer_name, max_length=256):
        if isinstance(file_paths, str):
            self.data = pd.read_csv(file_paths)
            
        else:
            self.data = pd.concat([pd.read_csv(file_path) for file_path in file_paths])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = 0 if self.data.iloc[idx]['label'] == 0 else 1
        
        encoding = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
    