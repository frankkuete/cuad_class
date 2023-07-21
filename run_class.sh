python run_seq_class.py \
--model_name "bert-base-uncased" \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 16 \
--train_file "./data/Non-Disparagement/train.csv" \
--test_file "./data/Non-Disparagement/test.csv" \

python run_seq_class.py \
--model_name "bert-base-uncased" \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 16 \
--train_file "./data/Non-Disparagement/oversampled-train.csv" \
--test_file "./data/Non-Disparagement/test.csv" \

python run_seq_class.py \
--model_name "bert-base-uncased" \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 16 \
--train_file "./data/Non-Disparagement/augmented-train.csv" \
--test_file "./data/Non-Disparagement/test.csv" \