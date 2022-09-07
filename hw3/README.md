predict your file
```
bash ./download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

my training script
```
CUDA_VISIBLE_DEVICES=$1 \
python3 run_summarization.py \
    --source_prefix "summarize: " \
    --text_column maintext \
    --summary_column title \
    --model_name_or_path mt5_load \
    --output_dir ./mt5_outputs \
    --train_file ./data/train.json \
    --do_train \
    --num_train_epochs 32 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_strategy steps \
    --logging_steps 1000 \
    --save_strategy steps \
    --save_steps 5000 \
    --seed 108 \
    --overwrite_output_dir
```

my predicting script
```
CUDA_VISIBLE_DEVICES=$1 \
python3 run_summarization.py \
    --source_prefix "summarize: " \
    --text_column maintext \
    --summary_column title \
    --model_name_or_path ./mt5_outputs/checkpoint-100000 \
    --output_dir ./mt5_predict/nogreedy \
    --test_file ./data/public.json \
    --train_file ./data/public.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_predict \
    --max_target_length 30 \
    --num_beams 10 \
    --seed 108 \
    --predict_with_generate \
    --overwrite_output_dir
```