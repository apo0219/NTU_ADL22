mkdir qa_load
mkdir mc_load
wget https://www.dropbox.com/s/8qhbalb56mprv6j/config.json?dl=1 -O qa_load/config.json
wget https://www.dropbox.com/s/uk6y05961h80z0g/pytorch_model.bin?dl=1 -O qa_load/pytorch_model.bin
wget https://www.dropbox.com/s/9536xpi2ayz9wdm/special_tokens_map.json?dl=1 -O qa_load/special_tokens_map.json
wget https://www.dropbox.com/s/ovyhq9tt6ovwyge/tokenizer_config.json?dl=1 -O qa_load/tokenizer_config.json
wget https://www.dropbox.com/s/02ehkq69qooo3r1/tokenizer.json?dl=1 -O qa_load/tokenizer.json
wget https://www.dropbox.com/s/25ovaosky9lqrzr/vocab.txt?dl=1 -O qa_load/vocab.txt

wget https://www.dropbox.com/s/onkolj41xv4gjqj/config.json?dl=1 -O mc_load/config.json
wget https://www.dropbox.com/s/eyljj34stk4ox8e/pytorch_model.bin?dl=1 -O mc_load/pytorch_model.bin
wget https://www.dropbox.com/s/gd16i5wdk6sfuli/special_tokens_map.json?dl=1 -O mc_load/special_tokens_map.json
wget https://www.dropbox.com/s/xujs0husbla1mpw/tokenizer_config.json?dl=1 -O mc_load/tokenizer_config.json
wget https://www.dropbox.com/s/0br1l24g4fp0xrw/tokenizer.json?dl=1 -O mc_load/tokenizer.json
wget https://www.dropbox.com/s/jaanifq20o7ap8w/vocab.txt?dl=1 -O mc_load/vocab.txt
