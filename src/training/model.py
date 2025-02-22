import torch
import torch.nn as nn
from transformers import AutoModel

class ToxicClassifier(nn.Module):
    
    def __init__(self, model_name: str, num_lables: int = 2):
        super(ToxicClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_lables)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        output = outputs.pooler_output # [CLS]
        output = self.dropout(output)
        return self.classifier(output)