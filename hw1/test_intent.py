from imaplib import Internaldate2tuple
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

from dataset import SeqClsDataset
from utils import Vocab
import torch
from torch.utils.data import DataLoader
from torch.nn import Softmax
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model import Intent_cls
import csv

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    idx2intent = {v:k for k,v in intent2idx.items()}
    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset, 1, drop_last = False)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model = Intent_cls(args.input_size, args.hidden_size, args.dropout, 150, 1, args.num_layers, args.model_type)
    model.eval()
    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(torch.load(args.ckpt_path))
    # TODO: predict dataset
    outdata = []
    for batch in dataloader:
        id = batch["id"][0]
        text = batch["text"][0].split(' ')
        x = []
        put = 0
        for j in text:
            if (j == ''):
                continue
            if (j in vocab.token2idx):
                x.append(embeddings[vocab.token2idx[j]])
                put += 1
            else:
                continue
        if (put == 0):
            outdata.append([id,'bill_balance'])
            continue
        l = x.__len__()
        x = torch.stack(x)
        padded_x = pad_sequence([x], batch_first = True)
        pack_padded_x = pack_padded_sequence(padded_x, [l], batch_first = True)
        y = model(pack_padded_x)
        outdata.append([id, idx2intent[y[0].argmax().item()]])
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file , 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','intent'])
        for i in outdata:
            writer.writerow(i)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json"
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/lstm.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--input_size", type=int, default = 300)
    parser.add_argument("--model_type", type=int, default=2)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
