# Prediction Reproducibility
```
bash ./download.sh
bash ./run.sh /path/to/context.json /path/to/test.json  /path/to/pred/prediction.csv
```
# Training reproducibility
## data preprocessing
```
// train file
mc_dataset_csv_gen.py /path/to/train.json /path/to/context.json /path/to/mc_train.csv
qa_dataset_csv_gen.py /path/to/train.json /path/to/context.json /path/to/qa_train.csv

//evaluate file
mc_dataset_csv_gen.py /path/to/valid.json /path/to/context.json /path/to/mc_valid.csv
qa_dataset_csv_gen.py /path/to/valid.json /path/to/context.json /path/to/qa_valid.csv
```
## train model:
Here is how I train the model. You can use `--help` to check the usage of the arguments for `run_swag.py` and `run_qa.py`.
Make sure you have the correct path for all the models, configs, ... , and files.
* swag :
`bash ./run_swag.sh cuda_device`
run_swag_sh:
    ```
    CUDA_VISIBLE_DEVICES=$1 python run_swag.py \
    --model_name_or_path ./roberta_load/pytorch_model.bin \
    --config_name ./roberta_load/config.json \
    --tokenizer_name ./roberta_load \
    --output_dir mc_roberta_out \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 2  \
    --max_seq_length 384 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --train_file mc_train.csv \
    --validation_file mc_valid.csv \
    ```
* qa:
`bash ./run_qa.sh cuda_device`
run_qa.sh:
    ```
    CUDA_VISIBLE_DEVICES=$1 python run_swag.py \
    --model_name_or_path ./roberta_load/pytorch_model.bin \
    --config_name ./roberta_load/config.json \
    --tokenizer_name ./roberta_load \
    --output_dir qa_roberta_out \
    --per_gpu_train_batch_size 1 \
    --gradient_accumulation_steps 2  \
    --max_seq_length 384 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --overwrite_output_dir \
    --train_file qa_train.csv \
    --validation_file qa_valid.csv \
    ```