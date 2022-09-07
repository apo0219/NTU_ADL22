python3 mc_test_csv_gen.py "${2}" "${1}"

CUDA_VISIBLE_DEVICES=0 python3 run_swag.py \
--model_name_or_path ./mc_load/pytorch_model.bin \
--config_name ./mc_load/config.json \
--tokenizer_name ./mc_load \
--output_dir ./mc_out \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--max_seq_length 384 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--do_predict \
--test_file mc_test.csv \
--overwrite_output_dir \
--output_file ./mc_out/out.txt

python3 qa_test_csv_gen.py "${2}" "${1}"

CUDA_VISIBLE_DEVICES=0 python run_qa.py \
--model_name_or_path ./qa_load/pytorch_model.bin \
--config_name ./qa_load/config.json \
--tokenizer_name ./qa_load \
--output_dir ./qa_out \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--max_seq_length 384 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--do_predict \
--test_file qa_test.csv \
--overwrite_output_dir \
                 
python3 output_json2csv.py "${3}"
