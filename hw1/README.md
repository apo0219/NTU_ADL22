# Writer
B09902128 黃宏鈺

# Please download the data first.
./download.sh

# reproduce the best model
./intent_cls.sh path1 path2
./slot_tag.sh path1 path2
* path1 : path to the testing file (.json)
* path2 : path to the output predictions (.csv)

# reproduce training
python3.8 train_intent.py --num_layer 2 --dropout 0.4 --hidden_size 512 --model_type 1 --num_epoch 200 --batch_size 512 --save_name model.pt --data_dir {where you put the training data}
python train_slot.py --num_layers 3 --dropout 0.5 --hidden_size 256 --model_type 2 --num_epoch 150 --batch_size 128 --save_name model.pt --data_dir {where you put the training data}