import yaml
from dataset import ToxicDataset
from model import ToxicClassifier
from trainer import train_model

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
# Load dataset
train_dataset = ToxicDataset(config["data"]["train_file"], config["model"]["name"], config["training"]["max_length"])
val_dataset = ToxicDataset(config["data"]["val_file"], config["model"]["name"], config["training"]["max_length"])

# Load model
model = ToxicClassifier(config["model"]["name"], config["model"]["num_labels"])

# Train model
train_model(model, train_dataset, val_dataset, config)
