# 🚀 Toxic Chat Classifier - Detecting Harmful Content

## 📌 Introduction
Toxic Chat Classifier is a machine learning model that leverages **BERT multilingual (`google-bert/bert-base-multilingual-cased`)** to detect toxic content in text messages.  
The system supports **fine-tuning, text prediction**, and a **RESTful API** for easy integration into applications.

---

## 🏗 **Project Structure**
```bash
toxic-chat-classifier/
│── configs/                     # Configuration files
│   ├── config.yaml               # Main config file
│── data/                         # Training dataset
│   ├── raw/                      # Raw datasets
│   ├── processed/                 # Processed datasets
│   ├── dataset_loader.py          # Dataset preprocessing and loading
│── models/                       # Model checkpoints
│   ├── pre_trained/               # Pre-trained models
│   ├── fine_tuned/                # Fine-tuned models
│── src/                          # Source code
│   ├── training/                  # Model fine-tuning scripts
│   │   ├── train.py                # Main training script
│   │   ├── dataset.py              # Dataset handling
│   │   ├── model.py                # Model definition
│   │   ├── trainer.py              # Training loop
│   ├── inference/                  # Inference scripts
│   │   ├── predict.py               # Inference function
│   ├── api/                        # FastAPI server for inference
│   │   ├── app.py                   # API routes
│── notebooks/                     # Jupyter Notebooks for testing
│   ├── data_exploration.ipynb      # Dataset analysis
│   ├── model_evaluation.ipynb      # Model evaluation
│── scripts/                       # Utility scripts
│   ├── preprocess.py               # Data preprocessing
│   ├── evaluate.py                 # Model evaluation
│   ├── test_api.py                 # API testing
│── tests/                         # Unit tests
│   ├── test_model.py                # Model inference tests
│   ├── test_api.py                  # API response tests
│── requirements.txt                # Dependencies
│── Dockerfile                      # Docker container setup
│── README.md                       # Documentation
│── .gitignore                       # Git ignore file
│── setup.py                         # Package installation script
```

---

## ⚙️ **Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your_username/toxic-chat-classifier.git
cd toxic-chat-classifier
```

### **2️⃣ Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Download Pre-trained Model (Optional)**
To fine-tune the model, you may need to download **`google-bert/bert-base-multilingual-cased`**:
```bash
from transformers import AutoModel
AutoModel.from_pretrained("google-bert/bert-base-multilingual-cased")
```

---

## 🎯 **Fine-Tuning the Model**
1. **Prepare your dataset** (CSV format with `"text"` and `"label"` columns).
2. **Modify `config.yaml`** in `configs/` with dataset paths.
3. **Run the training script**:
```bash
python src/training/train.py
```

### **🛠 Example Config (`configs/config.yaml`)**
```yaml
model:
  name: "google-bert/bert-base-multilingual-cased"
  num_labels: 2
  cache_dir: "models/"

training:
  batch_size: 16
  num_epochs: 3
  learning_rate: 5e-5
  weight_decay: 0.01
  max_length: 256
  warmup_steps: 500
  save_model_path: "models/fine_tuned/"

data:
  train_file: "data/processed/train.csv"
  val_file: "data/processed/val.csv"
  test_file: "data/processed/test.csv"
```

After fine-tuning, the model will be saved at `models/fine_tuned/`.

---

## 🚀 **Running the API**
### **1️⃣ Start the API Server**
Run the FastAPI server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### **2️⃣ Test the API**
**Swagger UI (Interactive API Documentation):**
- Open in browser: `http://127.0.0.1:8000/docs`

---

## 📌 **Using the API**
### **1️⃣ Predict a Single Text**
**Request:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
        "text": "You are an idiot!"
      }'
```
**Response:**
```json
{
  "text": "You are an idiot!",
  "label": "toxic",
  "confidence": 0.97
}
```

<!-- ---

## 🐳 **Deploying with Docker**
### **1️⃣ Build Docker Image**
```bash
docker build -t toxic-chat-classifier .
```

### **2️⃣ Run Docker Container**
```bash
docker run -p 8000:8000 toxic-chat-classifier
```
Now the API will be available at `http://127.0.0.1:8000`. -->

<!-- ---

## 🧪 **Running Tests**
To ensure everything works correctly, run unit tests:
```bash
pytest tests/
``` -->

---

## 📞 **Contact**
📧 Email: `Chungcgcg@gmail.com`  
