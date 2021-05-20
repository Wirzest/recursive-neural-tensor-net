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

stoi = pickle.load(open('./assets/stoi.pkl', 'rb'))

lexis_size = len(stoi)

BATCH_SIZE = 128
PARAMETERS = "./assets/batch_parameters/net_parameters_6.pth"

test = SSTDataset("./sst/test.txt", stoi)
testloader = DataLoader(test, batch_size=BATCH_SIZE)
N = test.__len__()

# Since Sentiment Tree have no support for GPU allocation
# they can't be fed to the model using cuda device. Training is done
# on the CPU with a subset of the training set.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = rntn.RNTensorN(lexis_size)
net.load_state_dict(torch.load(PARAMETERS))

test_loss = acc = 0

with torch.no_grad():
    for j, trees in enumerate(testloader, 0):
        for tree in trees:
            sentiment_tree = SentimentTree(tree, stoi, device)
            logits = net(sentiment_tree.root)
            lb = sentiment_tree.get_labels()
            ground_truth = torch.tensor([lb])[0]
            loss = net.get_loss(logits, ground_truth)
            acc += net.tree_accuracy.item()
            test_loss += loss.item()

acc /= N
test_loss /= N

print("Test loss: %.6f, Test accuracy: %.6f" % (test_loss, acc))
