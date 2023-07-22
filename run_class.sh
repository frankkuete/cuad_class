 #!/bin/bash

# List of data files
TRAINFILES=(
    "train.csv"
    "oversampled-train.csv"
    "augmented-train.csv"
)

MODELS=(
    "bert-base-uncased"
    "nlpaueb/legal-bert-base-uncased"
)

# Loop over each data file and execute the script
CATEGORIES=('Warranty Duration' 'Non-Disparagement' 'Post-Termination Services' 'Third Party Beneficiary')
 
for category in "${CATEGORIES[@]}"
do
  for model in "${MODELS[@]}"
  do
    for file in "${TRAINFILES[@]}"
    do
        python run_seq_class.py \
        --model_name "$model" \
        --learning_rate 1e-5 \
        --num_train_epochs 4 \
        --per_device_train_batch_size 24 \
        --per_device_eval_batch_size 16 \
        --train_file "./data/$category/$file" \
        --validation_file "./data/$category/validation.csv" \
        --test_file "./data/$category/test.csv" \
        
    done
  done
done