import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import shuffle
from struct import unpack
from tkinter import E
from typing import Dict
from xmlrpc.client import UNSUPPORTED_ENCODING
import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from torch.nn import RNN
from torch.nn import LSTM
from torch.nn import GRU
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import Softmax
from model import Slot_tag
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
RNN_ID = 0
LSTM_ID = 1
GRU_ID = 2

def maskLoss(input, target, mask):
    Total = mask.sum()
    loss = -torch.log(torch.gather(input, 2, target).squeeze())
    loss = loss.masked_select(mask).sum()
    return loss, Total.item()

def train(model, optimizer, x, y, mask):
    optimizer.zero_grad()
    predictions = model(x)
    loss, word_num = maskLoss(predictions, y, mask)
    loss.backward()
    optimizer.step()
    acc_nums = 0
    size = predictions.size()
    for i in range(size[0]):
        acc_tags = 0
        for j in range(size[1]):
            if (mask[i][j]):
                if (predictions[i][j].argmax() == y[i][j][0]):
                    acc_tags = acc_tags + 1
            else:
                acc_tags = acc_tags + 1            
        if (acc_tags == size[1]):
            acc_nums += 1
    return loss.item(), acc_nums

def evaluate(model, x, y, mask):
    predictions = model(x)
    loss, word_num = maskLoss(predictions, y, mask)
    acc_nums = 0
    size = predictions.size()
    for i in range(size[0]):
        acc_tags = 0
        for j in range(size[1]):
            if (mask[i][j]):
                if (predictions[i][j].argmax() == y[i][j][0]):
                    acc_tags = acc_tags + 1
            else:
                acc_tags = acc_tags + 1            
        if (acc_tags == size[1]):
            acc_nums += 1
    return loss.item(), acc_nums

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    for i in datasets[TRAIN]:
        i['tokens'] = " ".join(i['tokens'])
        i['tags'] = " ".join(i['tags'])
    for i in datasets[DEV]:
        i['tokens'] = " ".join(i['tokens'])
        i['tags'] = " ".join(i['tags'])        
    # TODO: crecate DataLoader for train / dev datasets
    BATCH_SIZE = args.batch_size
    dataloader = {split: DataLoader(datasets[split], batch_size = BATCH_SIZE, drop_last = True, shuffle = True) for split in SPLITS}
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    
    # TODO: init model and move model to target device(cpu / gpu)
    model = Slot_tag(args.input_size, args.hidden_size, args.dropout, 9, BATCH_SIZE, args.num_layers, GRU_ID)
    model.cuda(args.device)
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, eps = 1e-4)
    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    def batch2tensor(batch):
        Data = []
        for i in range(BATCH_SIZE):
            text_lst = batch['tokens'][i].split(' ')
            tag_lst = batch['tags'][i].split(' ')
            data = []
            for j in text_lst:
                data.append(embeddings[vocab.token2idx[j]])
            y = []
            for j in range(text_lst.__len__()):
                lab = [ tag2idx[tag_lst[j]] ]
                y.append(lab)
            y = torch.LongTensor(y)
            data.append(y)
            Data.append(data)
        Data.sort(key = len, reverse=True)
        X = []
        len_seq = []
        Y = []
        for i in Data:
            X.append(torch.stack(i[:-1]))
            len_seq.append(i.__len__() - 1)
            Y.append(i[-1])
        padded_x = pad_sequence(X, batch_first = True)
        pack_padded_x = pack_padded_sequence(padded_x, len_seq, batch_first = True)
        Y = pad_sequence(Y, batch_first = True)
        msk = []
        for i in len_seq:
            m = []
            for j in range(len_seq[0]):
                if (j < i):
                    m.append(1)
                else:
                    m.append(0)
            msk.append(m)
        msk = torch.BoolTensor(msk)
        return pack_padded_x, Y, msk
        
    best_acc = 0
    for epoch in epoch_pbar:
        train_loss = 0
        train_acc = 0
        model.train()
        count = 0
        for batch in dataloader[TRAIN]:
            x, y, msk = batch2tensor(batch)
            x = x.cuda(args.device)
            y = y.cuda(args.device)
            msk = msk.cuda(args.device)
            tmp_loss, tmp_acc = train(model, optimizer, x, y, msk)
            train_loss += tmp_loss
            train_acc += tmp_acc
            count+=BATCH_SIZE
        print("train loss = ", train_loss / count)
        print("train acc  = ", train_acc,'/',count,"=", train_acc / count)

        eval_loss = 0
        eval_acc = 0
        model.eval()
        count = 0
        with torch.no_grad():
            for batch in dataloader[DEV]:
                x, y, msk = batch2tensor(batch)
                x = x.cuda(args.device)
                y = y.cuda(args.device)
                msk = msk.cuda(args.device)
                tmp_loss, tmp_acc = evaluate(model, x, y, msk)
                eval_loss += tmp_loss
                eval_acc += tmp_acc
                count += BATCH_SIZE
        print("eval loss = ", eval_loss / count)
        print("eval acc  = ", eval_acc / count)

        if (eval_acc / count > best_acc):
            best_acc = eval_acc / count
            torch.save(model.state_dict(), str(args.ckpt_dir) + "/" + args.save_name + ".pt")
    
    print("final best : ", best_acc)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--input_size", type=int, default = 300)
    parser.add_argument("--save_name", type=str, default="best")
    parser.add_argument("--model_type", type=int, default=2)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=int, help="cpu, cuda, cuda:0, cuda:1", default = 0
    )
    parser.add_argument("--num_epoch", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
