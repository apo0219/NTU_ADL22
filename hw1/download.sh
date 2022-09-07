mkdir ckpt
mkdir cache
mkdir ckpt/intent
mkdir ckpt/slot
mkdir cache/intent
mkdir cache/slot
wget https://www.dropbox.com/s/84atrffioc96d9g/intent_cls.pt?dl=1 -O ./ckpt/intent/model.pt
wget https://www.dropbox.com/s/zn07w52zxkcxoct/intent_embeddings.pt?dl=1 -O ./cache/intent/embeddings.pt
wget https://www.dropbox.com/s/kc9g9go9tbpej8f/intent_vocab.pkl?dl=1 -O ./cache/intent/vocab.pkl
wget https://www.dropbox.com/s/ix9paxao805jxtk/intent2idx.json?dl=1 -O ./cache/intent/intent2idx.json
wget https://www.dropbox.com/s/1u2zc9i33z11jnn/slot_cls.pt?dl=1 -O ./ckpt/slot/model.pt
wget https://www.dropbox.com/s/6xc02nynrvhx5v6/slot_embeddings.pt?dl=1 -O ./cache/slot/embeddings.pt
wget https://www.dropbox.com/s/4szveu4k1youl5h/slot_vocab.pkl?dl=1 -O ./cache/slot/vocab.pkl
wget https://www.dropbox.com/s/a7eafw5euol8tsp/tag2idx.json?dl=1 -O ./cache/slot/tag2idx.json
