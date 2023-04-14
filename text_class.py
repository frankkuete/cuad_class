from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import json
import evaluate
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


loaded_dataset = load_dataset(
    'json', data_files='cuad_train.json', split="train")
loaded_test_set = load_dataset(
    'json', data_files='cuad_test.json', split="train")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_cuad_train = loaded_dataset.map(preprocess_function, batched=True)
tokenized_cuad_test = loaded_test_set.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="distilbert-cuad",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_cuad_train,
    eval_dataset=tokenized_cuad_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_cuad_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
