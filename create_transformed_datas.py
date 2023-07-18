from datasets import concatenate_datasets, load_dataset
import random
from tqdm import tqdm
import os
import csv
import nlpaug.augmenter.word as naw


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

def synonym_replacement(train_dataset):
    # Calculate the number of positive and negative samples in the training set
    pos_samples = train_dataset.filter(
        lambda example: example['label'] == 1).num_rows
    neg_samples = train_dataset.filter(
        lambda example: example['label'] == 0).num_rows

    positive_examples = train_dataset.filter(
        lambda example: example['label'] == 1)
    positive_examples = [example['text'] for example in positive_examples]
    augmented_dataset = [example for example in train_dataset]

    if neg_samples > pos_samples:
        offset = abs(pos_samples-neg_samples)
    else:
        return train_dataset

    for i in tqdm(list(range(offset))):
        prob = random.choice([0.2,0.4,0.6,0.8,0.99])
        clause_text = random.choice(positive_examples)
        result = naw.SynonymAug(
            aug_src='wordnet', aug_min=0.2*len(clause_text.split()), aug_p=prob).augment(clause_text)[0]
        
        augmented_dataset.append({"text": result, "label": 1})
    
    for data in augmented_dataset:
        if data["label"]==1:
            n_pos +=1
        else:
            n_neg +=1
    
    print("Number of Positive examples", n_pos,
          "Number of Negative examples", n_neg)

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
    
    print("Number of Positive examples", train_dataset.filter(lambda example: example['label'] == 1).num_rows, "Number of Negative examples", train_dataset.filter(lambda example: example['label'] == 0).num_rows)
    
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
        aug_dataset = synonym_replacement(dataset)
        print("#"*100)
        print("Creation of the random oversampled "+cat+" train dataset")
        create_csv_file(ros_dataset, "data/"+cat, "oversampled-train.csv")
        print("#"*100)
        create_csv_file(aug_dataset, "data/"+cat, "augmented-train.csv")
