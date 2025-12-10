from data_processing import data_prep
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch 


MODEL_NAME = "distilbert-base-uncased" #"meta-llama/Llama-3.2-3B-Instruct" "roberta-large"
DATASET_PATH = "./data/blooms_taxonomy_dataset.csv"
OUTPUT_PATH = "../Fine-Tuned-Models/distilbert-base-uncased"

tokenized_dataset, level_to_id, id_to_level, tokenizer, level_weights = data_prep(MODEL_NAME, DATASET_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(level_to_id),
    id2label=id_to_level,
    label2id=level_to_id
)

# New class that inherits Trainer to override compute_loss
# Inspired by:
# https://discuss.huggingface.co/t/create-a-weighted-loss-function-to-handle-imbalance/138178/3
class TrainerWithWeight(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch=None):

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels =inputs.get("labels") 

        device = next(model.parameters()).device
        weights_on_device = level_weights.to(device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_on_device)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


training_arguments = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=3,                  
    per_device_train_batch_size=8,       
    per_device_eval_batch_size=8,        
    learning_rate=5e-5,
    #fp16=True
)

trainer = TrainerWithWeight( 
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()

print("-------Evaluating Model-------")
results = trainer.evaluate()

trainer.save_model("./Fine-Tuned-Models/distilbert-base-uncased")
tokenizer.save_pretrained("./Fine-Tuned-Models/distilbert-base-uncased")
print("Model is saved")

