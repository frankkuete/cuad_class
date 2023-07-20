python run_seq_class.py \
--model_name "bert-base-uncased" \
--learning_rate 2e-5 \
--num_train_epochs 5 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 16 \
--train_file "./data/Non-Disparagement/train.csv" \
--test_file "./data/Non-Disparagement/test.csv" \