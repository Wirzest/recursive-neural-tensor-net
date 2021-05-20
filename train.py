# -*- coding: utf-8 -*-
# author: Victor H. Wirz
# discipline: UNIRIO-tin0145
# prof.: Pedro Moura

import rntn
from sentiment_tensor import SentimentTree
import pickle
import torch
from dataset import SSTDataset
from torch.utils.data import DataLoader
# from torch.nn.utils import clip_grad_norm_

stoi = pickle.load(open('./assets/stoi.pkl', 'rb'))

lexis_size = len(stoi)

EPOCHES = 10
BATCH_SIZE = 64

dataset = SSTDataset("./sst/small.txt", stoi)
validation = SSTDataset("./sst/dev.txt", stoi)
trainloader = DataLoader(dataset, batch_size=BATCH_SIZE)
valid_loader = DataLoader(validation)
N = dataset.__len__()

# Since Sentiment Tree have no support for GPU allocation
# they can't be fed to the model using cuda device. Training is done
# on the CPU with a subset of the training set.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = rntn.RNTensorN(lexis_size, 0.1)

optimizer = torch.optim.SGD(net.parameters(),
                            lr=0.05, momentum=0.9, dampening=0.0)

for i in range(EPOCHES):
    print("Epoch %d/%d:" % (i+1, EPOCHES))
    valid_acc = acc = running_loss = 0

    for j, trees in enumerate(trainloader, 0):
        optimizer.zero_grad()

        for tree in trees:
            sentiment_tree = SentimentTree(tree, stoi, device)
            logits = net(sentiment_tree.root)
            lb = sentiment_tree.get_labels()
            ground_truth = torch.tensor([lb])[0]
            loss = net.get_loss(logits, ground_truth)
            acc += net.tree_accuracy.item()
            loss.backward()
            running_loss += loss.item()

        # clip_grad_norm_(net.parameters(), 5, norm_type=2.)
        optimizer.step()

    for j, trees in enumerate(valid_loader, 0):
        for tree in trees:
            sentiment_tree = SentimentTree(tree, stoi, device)
            logits = net(sentiment_tree.root)
            lb = sentiment_tree.get_labels()
            ground_truth = torch.tensor([lb])[0]
            valid_acc += net.tree_accuracy.item()

    acc /= N
    running_loss /= N
    valid_acc /= validation.__len__()
    print("Loss: %.6f, Training accuracy: %.6f, Validation accuracy: %.6f" % (running_loss, acc, valid_acc))
    torch.save(net.state_dict(), "./assets/net_parameters_%d.pth" % i)
