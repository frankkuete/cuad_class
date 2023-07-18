from datasets import concatenate_datasets, load_dataset
import random
from tqdm import tqdm
import os
import csv
from BackTranslation import BackTranslation

def create_csv_file(data, file_path, filename):
    texts = [d['text'] for d in data]
    labels = [d['label'] for d in data]
    # Combine the texts and labels into a list of tuples
    tuples = list(zip(texts, labels))

    # Write the data to the CSV file
    with open(file_path+"/"+filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row (optional)
        writer.writerow(['text', 'label'])
        # Write each row of data
        writer.writerows(tuples)


def back_translation(train_dataset):
    trans = BackTranslation(url=[
        'translate.google.com',
        'translate.google.co.kr',
    ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})

    # Calculate the number of positive and negative samples in the training set
    pos_samples = train_dataset.filter(
        lambda example: example['label'] == 1).num_rows
    neg_samples = train_dataset.filter(
        lambda example: example['label'] == 0).num_rows

    positive_examples = train_dataset.filter(
        lambda example: example['label'] == 1)
    positive_examples = [example for example in positive_examples]
    augmented_dataset = [example for example in train_dataset]
    if neg_samples > pos_samples:
        offset = abs(pos_samples-neg_samples)//2
    else:
        return train_dataset

    for i in tqdm(list(range(offset//pos_samples))):
        for example in positive_examples:
            middle = random.choice(
                ["fr", "es", "nl", "it", "de"])
            result = trans.translate(
                example['text'], src='en', tmp=middle, sleeping=0.1)
            augmented_dataset.append({"text": result.result_text, "label": 1})
    return augmented_dataset

def random_oversampling(train_dataset):
    # Calculate the number of positive and negative samples in the training set
    pos_samples = train_dataset.filter(
        lambda example: example['label'] == 1).num_rows
    neg_samples = train_dataset.filter(
        lambda example: example['label'] == 0).num_rows
    # Oversample the positive samples
    positive_examples = train_dataset.filter(
        lambda example: example['label'] == 1)
    if neg_samples > pos_samples:
        offset = abs(pos_samples-neg_samples)
    else:
        return train_dataset
    for i in tqdm(list(range(offset//10))):
        positive_example = positive_examples.select(
            random.sample(range(0, pos_samples), 10))
        train_dataset = concatenate_datasets([train_dataset, positive_example])
    return train_dataset

if __name__ == "__main__":
    categories = ["Warranty Duration", "Non-Disparagement",
                  "Post-Termination Services", "Third Party Beneficiary"]
    if not os.path.exists("./data"):
        os.mkdir("./data")
    for cat in categories:
        if not os.path.exists("./data/"+cat):
            os.mkdir("./data/"+cat)
        
        dataset = load_dataset(
            "csv", data_files="./data/"+cat+"/train.csv", split="train")
        ros_dataset = random_oversampling(dataset)
        ros_dataset = [data for data in ros_dataset]
        aug_dataset = back_translation(dataset)
        print("#"*100)
        print("Creation of the random oversampled "+cat+" train dataset")
        create_csv_file(ros_dataset, "data/"+cat, "oversampled-train.csv")
        create_csv_file(aug_dataset, "data/"+cat, "augmented-train.csv")
