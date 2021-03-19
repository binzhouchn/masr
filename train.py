import torch
import torch.nn as nn
import data
from models.conv import GatedConv
from tqdm import tqdm
from decoder import GreedyDecoder
from warpctc_pytorch import CTCLoss
import torch.nn.functional as F
import joblib
from config import TRAIN_PATH, DEV_PATH, LABEL_PATH

import os
gpu_list = '2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

def train(
    model,
    epochs=110,
    batch_size=128,
    train_index_path=TRAIN_PATH,
    dev_index_path=DEV_PATH,
    labels_path=LABEL_PATH,
    learning_rate=0.6,
    momentum=0.8,
    max_grad_norm=0.2,
    weight_decay=0,
):
    train_dataset = data.MASRDataset(train_index_path, labels_path)
    batchs = (len(train_dataset) + batch_size - 1) // batch_size
    dev_dataset = data.MASRDataset(dev_index_path, labels_path)
    train_dataloader = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=0
    )
    train_dataloader_shuffle = data.MASRDataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    dev_dataloader = data.MASRDataLoader(
        dev_dataset, batch_size=batch_size, num_workers=0
    )
    parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=momentum,
        nesterov=True,
        weight_decay=weight_decay,
    )
    ctcloss = CTCLoss(size_average=True)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)

    gstep = 0
    for epoch in range(epochs):
        epoch_loss = 0
        if epoch > 0:
            train_dataloader = train_dataloader_shuffle
        # lr_sched.step()
        for i, (x, y, x_lens, y_lens) in enumerate(train_dataloader):
            x = x.cuda()
            out, out_lens = model(x, x_lens)
            out = out.transpose(0, 1).transpose(0, 2)
            loss = ctcloss(out, y, out_lens, y_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()
            gstep += 1
            print(
                "[{}/{}][{}/{}]\tLoss = {}".format(
                    epoch + 1, epochs, i, int(batchs), loss.item()
                )
            )
        epoch_loss = epoch_loss / batchs
        cer = eval(model, dev_dataloader)
        print("Epoch {}: Loss= {}, CER = {}".format(epoch, epoch_loss, cer))
        if (epoch+1) % 5 == 0:
            torch.save(model, "pretrained/model_{}.pth".format(epoch))

def eval(model, dataloader):
    model.eval()
    decoder = GreedyDecoder(dataloader.dataset.labels_str)
    cer = 0
    print("decoding")
    with torch.no_grad():
        for i, (x, y, x_lens, y_lens) in tqdm(enumerate(dataloader)):
            x = x.cuda()
            outs, out_lens = model(x, x_lens)
            outs = F.softmax(outs, 1)
            outs = outs.transpose(1, 2)
            ys = []
            offset = 0
            for y_len in y_lens:
                ys.append(y[offset : offset + y_len])
                offset += y_len
            out_strings, out_offsets = decoder.decode(outs, out_lens)
            y_strings = decoder.convert_to_strings(ys)
            for pred, truth in zip(out_strings, y_strings):
                trans, ref = pred[0], truth[0]
                cer += decoder.cer(trans, ref) / float(len(ref))
        cer /= len(dataloader.dataset)
    model.train()
    return cer


if __name__ == "__main__":
    vocabulary = joblib.load(LABEL_PATH)
    vocabulary = "".join(vocabulary)
    model = GatedConv(vocabulary)
    model.cuda()
    train(model)
