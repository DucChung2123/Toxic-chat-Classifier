import torch
from transformers import AutoTokenizer
from src.training.model import ToxicClassifier
import yaml

class ToxicClassifierAPI:
    
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ToxicClassifier(
            self.config["model"]["name"], num_labels=self.config["model"]["num_labels"]
        )
        self.model.load_state_dict(torch.load(
            self.config["training"]["save_model_path"] + ".pth",
            map_location=self.device
        ))
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])
        
    def predict(self, text: str):
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config["training"]["max_length"],
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()

        return {
            "text": text,
            "label": "toxic" if predicted_class == 1 else "normal",
            "confidence": probs[0, predicted_class].item()
        }
    
    def batch_predict(self, texts: list[str], batch_size: int = 32):
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.config["training"]["max_length"],
                return_tensors="pt",
                return_attention_mask=True
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probs, dim=1)

            for text, predicted_class, prob in zip(batch_texts, predicted_classes, probs):
                results.append({
                    "text": text,
                    "label": "toxic" if predicted_class == 1 else "normal",
                    "confidence": prob[predicted_class].item()
                })

        return results