import pandas as pd
import yaml
import requests
import time
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
test_df = pd.concat([pd.read_csv(path) for path in config["data"].get("test_file")])
# test_df = test_df[:200]
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

data_test = test_df["text"].tolist()
label_test = test_df["label"].tolist()

json_data = {
    'texts': data_test
}
print("-"*50)
print(f"Sending POST request to batch_predict endpoint")
print(f"Total test sample: {len(data_test)}")
start_time = time.time()
response = requests.post('http://0.0.0.0:8000/batch_predict/', headers=headers, json=json_data)
total_time = time.time() - start_time
print(f"Time take: {total_time:.2f} seconds")

# cal acc
response_json = response.json()
total_acc = 0
for res, label in zip(response_json, label_test):
    pred_label = 1 if res["label"] == "toxic" else 0
    if pred_label == label:
        total_acc += 1
print(f"Total accuracy: {total_acc*100/len(label_test):.2f}%")