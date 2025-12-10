from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "/Users/raymondeichner/Downloads/BloomAI/Fine-Tuned-Models/roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

id_to_level = {
    0 : "Remember",
    1 : "Understand",
    2 : "Apply",
    3 : "Analyze",
    4 : "Evaluate", 
    5 : "Create"
}

def classify(level: str) -> str:
    inputs = tokenizer(level, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = id_to_level[predicted_id]

    return predicted_label


