from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import os

# 1. Choose a small, fast model (DistilBERT is standard for sentiment)
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
save_directory = "./model"

# 2. Download and export to ONNX
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. Save the model and tokenizer files
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

# 4. Rename the file to match your project expectation
os.rename(os.path.join(save_directory, "model.onnx"), 
          os.path.join(save_directory, "sentiment_model.onnx"))

print(f"Model exported successfully to {save_directory}/sentiment_model.onnx")