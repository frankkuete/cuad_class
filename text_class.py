import os
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_dataset, concatenate_datasets
import json
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, recall_score
from scipy.special import softmax
import re
import nlpaug.augmenter.word as naw


def extract_from_cuad(question_cat="Covenant Not To Sue", is_train_data=True):
    """
    This function will extract the context of every example where the question is related to the given the question category ,
    then it will split the context in several excerpt will assign the label 1 , it the excerpt contains partially or totally 
    the answer or 0 if not.
    """
    nltk.download('punkt')
    # Load the CUAD train set or the CUAD testset
    if is_train_data:
        cuad = load_dataset('cuad', split='train')
    else:
        cuad = load_dataset('cuad', split='test')

    # Initialize empty lists for the texts and labels
    texts = []
    labels = []
    for i in range(len(cuad)):
        if question_cat in cuad[i]['id'] and cuad[i]['answers']['text'] != []:
            context = ''.join(cuad[i]['context'])
            context_sentences = sent_tokenize(context)
            answers = []
            for answer in cuad[i]['answers']['text']:
                answers.append(answer)
            for sentence in context_sentences:
                texts.append(sentence)
                sent_exists = False
                for answer in answers:
                    if answer.find(sentence) != -1 or sentence.find(answer) != -1:
                        sent_exists = True
                if sentence in answers or sent_exists:
                    labels.append(1)
                else:
                    labels.append(0)
    return texts, labels


def main(category, tokenizer_name, model_name, train_batch_size, eval_batch_size, lr, num_epoch):

    ######################### DATA PREPROCESSING #########################
    train_texts, train_labels = extract_from_cuad(category)

    loaded_dataset = {
        'text': train_texts,
        'label': train_labels,
    }

    test_texts, test_labels = extract_from_cuad(category, False)

    loaded_test_set = {
        'text': test_texts,
        'label': test_labels,
    }

    # Convert train and validation datasets to Dataset objects from huggingface
    train_dataset = Dataset.from_dict(loaded_dataset)
    validation_dataset = Dataset.from_dict(loaded_test_set)

    # select small slices of the training , validation set
    """
    train_dataset = concatenate_datasets([train_dataset.filter(lambda example: example['label'] == 0).select(range(100)),
                                          train_dataset.filter(lambda example: example['label'] == 1).select(range(10))])
    validation_dataset = validation_dataset.select(range(2000))
    """
    ################################ DATA CLEANING ################################
    print("############### DATA CLEANING ###############")

    def replace_spaces(string):
        # Remove long spaces
        return re.sub(r' {2,}', ' ', string)

    def replace_emptyline(string):
        # Remove empty lines using regular expressions
        return re.sub(r'\n\s*\n|\n\s+', '\n', string)

    def pre_process_examples(example):
        return replace_spaces(replace_emptyline(example))

    def invalid_example(example):
        if example['label'] == 0:
            is_invalid = re.match('^(ARTICLE|Page|section|Section|\d|^\([A-Za-z]\))', example['text']) or re.search(
                r'\[\*\*\*\]', example['text']) or len(example['text'].split()) <= 10 or re.search(r'\.{2,}', example['text'])
        elif len(example['text'].split()) < 3:
            is_invalid = True
        else:
            is_invalid = False
        return not is_invalid

    train_dataset = train_dataset.map(
        lambda example: {'text': pre_process_examples(example['text'])})
    train_dataset = train_dataset.filter(invalid_example)

    validation_dataset = validation_dataset.map(
        lambda example: {'text': pre_process_examples(example['text'])})
    validation_dataset = validation_dataset.filter(invalid_example)

    print("train_dataset", train_dataset.num_rows, train_dataset.filter(
        lambda example: example['label'] == 1).num_rows)
    print("val_dataset", validation_dataset.num_rows, validation_dataset.filter(
        lambda example: example['label'] == 1).num_rows)
    ################################ METHOD 1 : DATA AUGMENTATION ####################################

    def data_augmentation(processed_train_dataset):
        # Doc: https://nlpaug.readthedocs.io/en/latest/augmenter/word/antonym.html
        # Get the number of examples in the minority class
        minority_class_ratio = processed_train_dataset.filter(
            lambda example: example['label'] == 1).num_rows/processed_train_dataset.num_rows*100

        print("***********Minority class ratio********\n ",
              round(minority_class_ratio, 3))

        n_positive, n_total = processed_train_dataset.filter(
            lambda example: example['label'] == 1).num_rows, processed_train_dataset.num_rows
        augmented_minority_class_examples = []
        for i in range((n_total-n_positive)//(n_positive*5)):
            # Generate augmented examples for the minority class
            minority_class_examples = processed_train_dataset.filter(
                lambda example: example['label'] == 1)['text']
            for example in minority_class_examples:
                wc = len(example.split())
                # Substitute word by WordNet's synonym
                augmented_minority_class_examples.append(naw.SynonymAug(
                    aug_src='wordnet', aug_min=0.2*wc, aug_max=wc, aug_p=0.2).augment(example)[0])
                augmented_minority_class_examples.append(naw.SynonymAug(
                    aug_src='wordnet', aug_min=0.4*wc, aug_max=wc, aug_p=0.4).augment(example)[0])
                augmented_minority_class_examples.append(naw.SynonymAug(
                    aug_src='wordnet', aug_min=0.6*wc, aug_max=wc, aug_p=0.6).augment(example)[0])
                augmented_minority_class_examples.append(naw.SynonymAug(
                    aug_src='wordnet', aug_min=0.8*wc, aug_max=wc, aug_p=0.8).augment(example)[0])
                augmented_minority_class_examples.append(naw.SynonymAug(
                    aug_src='wordnet', aug_min=0.99*wc, aug_max=wc, aug_p=0.99).augment(example)[0])

        # Combine the augmented examples with the original dataset
        augmented_minority_class_examples = {
            'text': augmented_minority_class_examples,
            'label': [1 for i in range(len(augmented_minority_class_examples))],
        }
        augmented_minority_dataset = Dataset.from_dict(
            augmented_minority_class_examples)
        processed_train_dataset = concatenate_datasets(
            [processed_train_dataset, augmented_minority_dataset])

        # Print the new class distribution
        minority_class_ratio = processed_train_dataset.filter(
            lambda example: example['label'] == 1).num_rows/processed_train_dataset.num_rows*100
        print("***********Minority class ratio********\n ", minority_class_ratio)
        return processed_train_dataset
    augmented_train_dataset = data_augmentation(train_dataset)

    ################################ METHOD 2 : DATA OVERSAMPLING ####################################

    def oversampling_dataset(train_dataset):
        # Calculate the number of positive and negative samples in the training set
        pos_samples = sum(train_dataset['label'])
        neg_samples = len(train_dataset) - pos_samples
        pos_samples, neg_samples
        # Oversample the positive samples
        oversampled_train_dataset = train_dataset
        positive_examples = train_dataset.filter(
            lambda example: example['label'] == 1)
        if pos_samples > 0:
            while pos_samples < neg_samples:
                oversampled_train_dataset = concatenate_datasets(
                    [oversampled_train_dataset, positive_examples])
                pos_samples += len(positive_examples)
        return oversampled_train_dataset
    oversampled_train_dataset = oversampling_dataset(train_dataset)
    ################################ TOKENIZATION ####################################

    # tokenized the the train , validation sets using the tokenizer from the appropriate model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_cuad_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_cuad_train_oversampled = oversampled_train_dataset.map(
        preprocess_function, batched=True)
    tokenized_cuad_train_augmented = augmented_train_dataset.map(
        preprocess_function, batched=True)
    tokenized_cuad_validation = validation_dataset.map(
        preprocess_function, batched=True)

    ################################ TRAIN ############################################

    training_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

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
        thresholds = np.linspace(0, 1, num=101)
        precisions = []
        recalls = []
        for threshold in thresholds:
            threshold_predictions = probs[:, 1] > threshold
            threshold_predictions = threshold_predictions.astype(int)
            precisions.append(precision_score(
                y_true=labels, y_pred=threshold_predictions, zero_division=1))
            recalls.append(recall_score(
                y_true=labels, y_pred=threshold_predictions, zero_division=1))

        # Compute AUPR
        aupr = auc(recalls, precisions)

        # Return results as a dictionary
        results = {"accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
                   "precision_at_80_recall":  get_prec_at_recall(precisions, recalls, thresholds, recall_thresh=0.8),
                   "precision_at_90_recall":  get_prec_at_recall(precisions, recalls, thresholds, recall_thresh=0.9),
                   "aupr": aupr}

        return results

    # Before you start training your model, create a map of the expected ids to their labels with id2label and label2id:
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # setup your model for SequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )
    # define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=training_tokenizer)

    datasets = [tokenized_cuad_train,
                tokenized_cuad_train_oversampled, tokenized_cuad_train_augmented]
    methods = ["vanilla", "oversampled", "augmented"]
    results = {}

    for dataset, method in zip(datasets, methods):
        # instantiate the training arguments
        training_args = TrainingArguments(
            output_dir="../models/{}-{}-cuad".format(model_name, method),
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epoch,
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
            train_dataset=dataset,
            eval_dataset=tokenized_cuad_validation,
            tokenizer=training_tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # start the training
        trainer.train()
        trainer.save_model()
        trainer.save_state()

        # Evaluate the model on the validation set
        eval_results = trainer.evaluate()
        results[method] = eval_results
        results["val_set_size"] = validation_dataset.num_rows 
        results["val_pos_size"] =validation_dataset.filter(lambda example: example['label'] == 1).num_rows
        # Print the evaluation results
        print(eval_results)

    # Save evaluation results to a JSON file
    save_dir = "./results"
    save_path = os.path.join(
        save_dir, "results_{}.json".format(category))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main(category="Competitive Restriction Exception", tokenizer_name="google/electra-large-discriminator",
         model_name="google/electra-large-discriminator", train_batch_size=8, eval_batch_size=32, lr=1e-5, num_epoch=1)
