from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets , Dataset
import nltk
from nltk.tokenize import sent_tokenize
import re
import csv
import os
nltk.download('punkt')   

CUAD=load_dataset('cuad')


def replace_spaces(string):
    # Remove long spaces 
    return re.sub(r' {2,}', ' ', string)
def replace_emptyline(string):
    # Remove empty lines using regular expressions
    return re.sub(r'\n\s*\n|\n\s+', '\n', string)

def preprocess_clause(clause_text):
        return replace_spaces( replace_emptyline (clause_text) )

def valid_example(clause_text):
    """
    this function checks if a clause is valid  
    """
    if re.match('^(ARTICLE|Page|page|article|section|Section|\d|^\([A-Za-z]\) )',clause_text):
        return False
    elif re.search(r'\[\*\*\*\]',clause_text):
        return False
    elif len(clause_text.split())<=25 or len(clause_text.split())>=200:
        return False
    elif re.search(r'\.{2,}',clause_text):
        return False
    elif re.search(r'\-{2,}',clause_text):
        return False
    else:
        return True

def get_pos_relevant_examples(category ,data_split):
    """
    This function returns the CUAD positive examples which is related to given category in CUAD train dataset
    """
    pos_index = []
    pos_relevant_index = []
    if data_split == 'train':
        cuad = CUAD['train']
    else:
        cuad = CUAD['test']
    for i in range(len(cuad)):
        if cuad[i]['answers']['text'] != []:
            pos_index.append(i)
    for index in pos_index:
        if category in cuad[index]['question']:
            pos_relevant_index.append(index)
    return cuad.select(pos_relevant_index)

def extract_clauses(contracts):
    """
    this functions extracts all clauses from a given list of contracts
    """
    clauses = []
    for contract in contracts:
        for clause in sent_tokenize(contract):
            clauses.append(clause)
    return list(set(clauses))

def clauses_labelling(clauses,annotations):
    dataset=[]
    for clause in clauses:
        found=False
        for annotation in annotations:
            if clause.find(annotation)!=-1: 
                found=True
        if found:
            dataset.append({"text":clause,"label":1})
        else:
            dataset.append({"text":clause,"label":0})
    return dataset

def create_csv_file(data, file_path , filename ):
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
        
def create_dataset(category , data_split):
    relevant_pos_examples = get_pos_relevant_examples(category,data_split)
    
    # get unique contracts
    unique_contracts = list(set(relevant_pos_examples['context']))
    
    # get unique clauses from contracts
    clauses = extract_clauses(unique_contracts)
    
    # get the positive examples for the category
    cat_pos_examples = relevant_pos_examples.filter(lambda example: category in example["question"])
    
    # extract all the annotations for the category
    cat_annotations = []
    for ans in cat_pos_examples["answers"]:
        for text in ans['text']:
            cat_annotations.append(text)
    
    labeled_clauses = clauses_labelling(clauses , cat_annotations)
    # create labeled_clauses
    for labeled_clause in labeled_clauses:
        labeled_clause['text'] = preprocess_clause(labeled_clause['text'])
    # remove invalid negative clauses or very long positive clauses 
    labeled_clauses = [d for d in labeled_clauses if (valid_example(d['text']) and d['label']==0 ) or (len(d['text'].split())<500 and d['label']==1 ) ]
        
    return labeled_clauses

if __name__ == "__main__":

    categories = ["Warranty Duration", "Non-Disparagement",
                  "Post-Termination Services", "Third Party Beneficiary"]
    if not os.path.exists("./data"): 
            os.mkdir("./data")
    for cat in categories:  
        if not os.path.exists("./data/"+cat): 
            os.mkdir("./data/"+cat)
        
        print("#"*100)
        print("Creation of the "+cat+" datasets")

        # generate the train set 
        train_set = create_dataset(cat , "train") 

        # split the train set positive examples into train and validation set 
        pos_splitted_examples = Dataset.from_list(train_set).filter(
            lambda example: example['label'] == 1).train_test_split(test_size=0.1, shuffle=False)
        
        # split the train set negative examples into train and validation set
        neg_splitted_examples = Dataset.from_list(train_set).filter(
            lambda example: example['label'] == 0).train_test_split(test_size=0.1, shuffle=False)
        
        print("total numbers : ", Dataset.from_list(train_set).num_rows, "num pos train ",
              pos_splitted_examples['train'].num_rows, "num neg train", neg_splitted_examples['train'].num_rows, "num pos valid", pos_splitted_examples['test'].num_rows, "num neg valid ", neg_splitted_examples['test'].num_rows)

        # create final train set and validation set
        train_set = concatenate_datasets([pos_splitted_examples["train"], neg_splitted_examples["train"]])
        validation_set = concatenate_datasets(
            [pos_splitted_examples["test"], neg_splitted_examples["test"]])
        
        create_csv_file([example for example in train_set],
                        "data/" + cat ,  "/train.csv")
        create_csv_file([example for example in validation_set],
                        "data/" + cat ,  "validation.csv")

        # generate the test set
        test_set = create_dataset(cat, "test")
        create_csv_file( test_set, "data/"+ cat , "/test.csv")