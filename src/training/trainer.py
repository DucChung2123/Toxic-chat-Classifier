import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm  

def train_model(model, train_dataset, val_dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=config["training"].get("batch_size", 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"].get("batch_size", 32), shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config["training"].get("learning_rate", "5e-5")),
        weight_decay=config["training"].get("weight_decay", 0.01)
    )
    
    num_epochs = config["training"].get("num_epochs", 3)
    batch_size = config["training"].get("batch_size", 32)

    num_batches_per_epoch = len(train_dataset) // batch_size
    num_training_steps = num_epochs * num_batches_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config["training"].get("num_warmup_steps", 500),
        num_training_steps=num_training_steps
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=True)

        for batch in train_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_progress.set_postfix(loss=loss.item(), acc=correct/total)

        print(f"\nEpoch {epoch+1} Completed - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}\n")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=True)

        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Cập nhật progress bar cho validation
                val_progress.set_postfix(val_loss=loss.item(), val_acc=val_correct/val_total)

        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_correct/val_total:.4f}\n")
        
    # Lưu model sau khi train xong
    torch.save(model.state_dict(), config["training"]["save_model_path"] + "model_toxic_classifier.pth")
