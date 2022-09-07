python3 jsonl2json.py --filename $1
python3 run_summarization.py \
    --source_prefix "summarize: " \
    --text_column maintext \
    --summary_column title \
    --model_name_or_path ./mt5_load \
    --output_dir ./mt5_outputs \
    --test_file ./input.json \
    --train_file ./input.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_predict \
    --max_target_length 30 \
    --num_beams 10 \
    --seed 108 \
    --predict_with_generate \
    --overwrite_output_dir
python3 txt2json.py --dir ./mt5_outputs
mv ./mt5_outputs/predict.json $2