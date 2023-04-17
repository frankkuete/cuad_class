from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_dataset
import json
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split


def find_subtext_indices(text, subtext):
    """Finds the start and end indices of a subtext within a larger text."""
    # Find the starting index of the subtext in the text
    start_index = text.find(subtext)
    if start_index == -1:
        # Subtext not found in text
        return -1, -1
    else:
        # Find the ending index of the subtext in the text
        end_index = start_index + len(subtext)
        return start_index, end_index


def extract_from_cuad(text_max_length=256, question_cat="Covenant Not To Sue", model_name="bert-base-uncased", is_train_data=True):
    # Load the CUAD train set or the CUAD test set
    if is_train_data:
        cuad = load_dataset('cuad', split='train')
    else:
        cuad = load_dataset('cuad', split='test')

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the maximum length of the sequences
    max_length = text_max_length

    # Initialize empty lists for the texts and labels
    texts = []
    labels = []
    # for every answer related to the question category, we split the context into several texts
    # having at most max_length tokens , we label 1 the text if the text contains the anwser and 0 otherwise
    for i in range(len(cuad)):
        if question_cat in cuad[i]['id'] and cuad[i]['answers']['text'] != []:
            context = ''.join(cuad[i]['context'])
            answer = ''.join(cuad[i]['answers']['text'])
            encoded_context = tokenizer.encode(
                context, add_special_tokens=False)
            encoded_answer = tokenizer.encode(answer, add_special_tokens=False)
            processed_context = tokenizer.decode(encoded_context)
            processed_answer = tokenizer.decode(encoded_answer)
            ans_start_idx, ans_end_idx = find_subtext_indices(
                processed_context, processed_answer)
            spans = [encoded_context[i:i+max_length]
                     for i in range(0, len(encoded_context), max_length)]
            prev_start_idx, prev_end_idx = 0, len(tokenizer.decode(spans[0]))
            for span in spans:
                text = tokenizer.decode(span)
                texts.append(text)
                span_start_idx, span_end_idx = find_subtext_indices(
                    processed_context, text)
                if span_start_idx == -1:
                    span_start_idx, span_end_idx = prev_end_idx + \
                        1, prev_end_idx+1 + len(text)
                prev_start_idx, prev_end_idx = span_start_idx, span_end_idx
                up_intersect = (
                    span_start_idx <= ans_start_idx and ans_start_idx <= span_end_idx)
                down_intersect = (span_start_idx <=
                                  ans_end_idx and ans_end_idx <= span_end_idx)
                has_answer = int(up_intersect or down_intersect)
                labels.append(has_answer)
    return texts, labels


def main():
    # DATA PREPROCESSING
    train_texts, train_labels = extract_from_cuad(
        128, "Covenant Not To Sue", "bert-base-uncased")

    loaded_dataset = {
        'text': train_texts,
        'label': train_labels,
    }
    loaded_dataset = Dataset.from_dict(loaded_dataset)

    # Split the the cuad dataset into train and validation sets
    train_dataset, validation_dataset = train_test_split(
        loaded_dataset, test_size=0.2)

    test_texts, test_labels = extract_from_cuad(
        128, "Covenant Not To Sue", "bert-base-uncased", False)

    loaded_test_set = {
        'text': test_texts,
        'label': test_labels,
    }

    # Convert train and validation datasets to Dataset objects from huggingface
    train_dataset = Dataset.from_dict(train_dataset)
    validation_dataset = Dataset.from_dict(validation_dataset)
    test_dataset = Dataset.from_dict(loaded_test_set)

    print(len(train_dataset), len(validation_dataset), len(test_dataset))
    # tokenized the the train , validation and test set using the tokenizer from the appropriate model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_cuad_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_cuad_validation = validation_dataset.map(
        preprocess_function, batched=True)
    tokenized_cuad_test = test_dataset.map(
        preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = evaluate.load("accuracy")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # TRAIN

    # Evaluate the results of the training
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    # Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # setup your model for SequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="distilbert-cuad",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    # instantiate a trainer given the Training arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_cuad_train,
        eval_dataset=tokenized_cuad_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()
