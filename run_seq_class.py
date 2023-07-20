import argparse
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import concatenate_datasets, load_dataset
from scipy.special import softmax
import evaluate
import numpy as np 
from sklearn.metrics import auc, precision_score, recall_score
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Running sequence classification model')

    parser.add_argument(
        "--train_file", help="The input training data file (a text file).", type=str)
    
    parser.add_argument(
        "--test_file", help="The input test data file (a text file).",type=str)
    
    parser.add_argument("--model_name", help="Path to pretrained model or model identifier from huggingface.co/models")

    parser.add_argument( "--learning_rate", help="learning_rate of the during the training phase")


    parser.add_argument("--num_train_epochs", type=int, help="number of training epochs")

    parser.add_argument("--per_device_train_batch_size", type=int,
                        help=" number of examples processed together in parallel during one forward and backward pass(one training step)through the model.")
    
    parser.add_argument("--per_device_eval_batch_size", type=int,
                        help="number of examples processed together in parallel during one evaluation step.")

    args = parser.parse_args()

    print(args.train_file)
    print(args.model_name)
    print(args.learning_rate)
    print(args.num_train_epochs)
    print(args.per_device_train_batch_size)
    print(args.per_device_eval_batch_size)
    print(args.model_name)

    #----------------------------------------------------------------------------------------------------------#
    #                                                                                                          #
    #                                       Data preprocessing                                                 #
    #                                                                                                          #
    #----------------------------------------------------------------------------------------------------------#
    
    # Loading the training set and the test set from the given files
    train_set = load_dataset(
        "csv", data_files=args.train_file, split="train")
    test_set = load_dataset(
        "csv", data_files=args.test_file, split="train")
    
    pos_split_dataset = train_set.filter(
        lambda example: example['label'] == 1).train_test_split(test_size=0.1)
    neg_split_dataset = train_set.filter(
        lambda example: example['label'] == 0).train_test_split(test_size=0.1)
    
    train_set = concatenate_datasets(
        [pos_split_dataset['train'], neg_split_dataset['train']])
    validation_set = concatenate_datasets(
        [pos_split_dataset['test'], neg_split_dataset['test']])

    # Create a tokenizer for the specified model name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # Tokenize the train , validation and test datasets 
    tokenized_train_set = train_set.map(
        preprocess_function, batched=True)

    tokenized_test_set = test_set.map(
        preprocess_function, batched=True)
    
    tokenized_validation_set = validation_set.map(
        preprocess_function, batched=True)
    
    print("#"*15)
    print(tokenized_train_set.num_rows,tokenized_train_set.filter(
        lambda example: example['label'] == 1).num_rows, tokenized_train_set.filter(
        lambda example: example['label'] == 0).num_rows)
    print("#"*15)
    print(tokenized_test_set.num_rows, tokenized_test_set.filter(
        lambda example: example['label'] == 1).num_rows, tokenized_test_set.filter(
        lambda example: example['label'] == 0).num_rows)
    print("#"*15)
    print(tokenized_validation_set.num_rows, tokenized_validation_set.filter(
        lambda example: example['label'] == 1).num_rows, tokenized_validation_set.filter(
        lambda example: example['label'] == 0).num_rows)

    #----------------------------------------------------------------------------------------------------------#
    #                                                                                                          #
    #                                           Training                                                       #
    #                                                                                                          #
    #----------------------------------------------------------------------------------------------------------#
    training_tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Evaluate the results of the training
    accuracy = evaluate.load("accuracy")

    def get_prec_at_recall(precisions, recalls, confs, recall_thresh=0.9):
        """
        return the precisions , for a specified recall theshold
        Assumes recalls are sorted in increasing order
        """
        prec_at_recall = 0
        for prec, recall, conf in zip(reversed(precisions), reversed(recalls), confs):
            if recall >= recall_thresh:
                prec_at_recall = prec
                break
        return prec_at_recall

    def compute_metrics(eval_pred):
        # get predictions logits and gt_labels
        predictions, labels = eval_pred

        # Convert logits to probabilities using  softmax function
        probs = softmax(predictions, axis=1)

        predictions = np.argmax(predictions, axis=1)

        # precision-recall curve
        thresholds = np.linspace(0, 0.99, num=100)
        precisions = []
        recalls = []
        for threshold in thresholds:
            print(probs)
            threshold_predictions = probs[:, 1] > threshold
            print(threshold_predictions)
            threshold_predictions = threshold_predictions.astype(int)
            precisions.append(precision_score(
                y_true=labels, y_pred=threshold_predictions, zero_division=1))
            recalls.append(recall_score(
                y_true=labels, y_pred=threshold_predictions, zero_division=1))

        # Compute AUPR
        aupr = auc(recalls, precisions)

        # Return results as a dictionary
        results = {"accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
                   "precision": precisions,
                   "recall": recalls,
                   "precision_at_80_recall":  get_prec_at_recall(precisions, recalls, thresholds, recall_thresh=0.8),
                   "precision_at_90_recall":  get_prec_at_recall(precisions, recalls, thresholds, recall_thresh=0.9),
                   "aupr": aupr}

        return results

    # Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # setup your model for SequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label=id2label, label2id=label2id
    )
    # define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=training_tokenizer)

    # instantiate the training arguments
    training_args = TrainingArguments(
        output_dir="./",
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model='aupr',
        seed=42,
        push_to_hub=False,
    )
    
    # instantiate a trainer given the Training arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_validation_set,
        tokenizer=training_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # start the training
    #trainer.train()
    #trainer.save_model()
    #trainer.save_state()

    # validate the model on the validation set
    predictions = trainer.predict(test_dataset=tokenized_test_set)

    # Print the evaluation results
    #print(predictions.metrics)

    # Save evaluation results to a JSON file

