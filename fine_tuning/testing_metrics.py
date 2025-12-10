import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer 
from data_processing import data_prep

MODEL_NAME = "roberta-large"
MODEL_PATH = "/Users/raymondeichner/Downloads/BloomAI/Fine-Tuned-Models/roberta-large"
DATASET_PATH = "./data/blooms_taxonomy_dataset.csv"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenized_dataset, level_to_id, id_to_level, tokenizer, level_weights = data_prep(MODEL_PATH, DATASET_PATH)

def compute_metrics(eval_pred):

    logits, correct_level = eval_pred
    predicted_level = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    confusion_matrix = evaluate.load("confusion_matrix")

    accuracy_metric = accuracy.compute(predictions=predicted_level, references=correct_level)
    f1_metric = f1.compute(predictions=predicted_level, references=correct_level, average="macro")
    precision_metric = precision.compute(predictions=predicted_level, references=correct_level, average="weighted")
    recall_metric = recall.compute(predictions=predicted_level, references=correct_level, average="macro")
    confusion_matrix_metric = confusion_matrix.compute(predictions=predicted_level, references=correct_level)

    print("\n\n----------Model Metrics----------")
    print("Accuracy: ", accuracy_metric)
    print("F1: ", f1_metric)
    print("Precision: ", precision_metric)
    print("Recall: ", recall_metric)
    print("Confusion Matrix: ", confusion_matrix_metric)

    return {
        "accuracy" : accuracy_metric["accuracy"],
        "f1": f1_metric["f1"],
        "precision" : precision_metric["precision"],
        "recall" : recall_metric["recall"]
    }

trainer = Trainer(
    model=model,
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.evaluate()
