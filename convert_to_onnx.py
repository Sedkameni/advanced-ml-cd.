import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 1. Setup paths
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
save_path = "app/model"
os.makedirs(save_path, exist_ok=True)

# 2. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 3. Create dummy input for the ONNX exporter
dummy_input = tokenizer("This is a dummy input", return_tensors="pt")

# 4. Export to ONNX
torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    os.path.join(save_path, "sentiment_model.onnx"),
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=14
)

print(f"Model successfully saved to {save_path}/sentiment_model.onnx")