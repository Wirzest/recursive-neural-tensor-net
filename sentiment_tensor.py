# -*- coding: utf-8 -*-
# author: Victor H. Wirz
# discipline: UNIRIO-tin0145
# prof.: Pedro Moura

from nltk.tree import ParentedTree
import torch

_UNK = '<unk>'


class SentimentTree():
    """
    Hierarchical structure used to read and represent the
    Stanford Treebank.

    It uses the nltk implementation of fromstring() to catch the
    representation of syntax trees and morphological trees. But
    the ending object is not an instance of any nklt package classes.
    """
    def __init__(self, tree_raw, stoi, device):
        nltk_tree = ParentedTree.fromstring(tree_raw)

        for leaf_idx in nltk_tree.treepositions('leaves'):
            if nltk_tree[leaf_idx] in stoi:
                nltk_tree[leaf_idx] = stoi[nltk_tree[leaf_idx]]
            else:
                nltk_tree[leaf_idx] = stoi[_UNK]

        self.device = device
        self.root = self.parse(nltk_tree)
        self.labels = self._get_labels_(self.root)

    def parse(self, nltk_tree):
        """
            Nodes are tensors objects extended with additional
            pointers to left and right sides of the tree.

            Nodes cointain two entries:
                0-th: sentiment label
                1-st: word index.
                    (-1 is a dummy value used for non-leafs only)
        """
        if nltk_tree.height() == 2:
            node = torch.tensor([int(nltk_tree.label()), nltk_tree[0]], device=self.device)
            node.leaf = True
            node.left = None
            node.right = None
            return node
        else:
            node = torch.tensor([int(nltk_tree.label()), -1], device=self.device)
            node.leaf = False
            node.left = self.parse(nltk_tree[0])
            node.right = self.parse(nltk_tree[1])
            return node

    def _get_labels_(self, node):
        if node is None:
            return []

        return self._get_labels_(node.left) + self._get_labels_(node.right) + [node[0].item()]

    def get_labels(self):
        return self.labels


if __name__ == "__main__":
    import pickle

    tree_raw = """(3 (2 It) (4 (4 (2 's)
                (4 (3 (2 a) (4 (3 lovely)(2 film)))
                (3 (2 with) (4 (3 (3 lovely)(2 performances))
                (2 (2 by)(2 (2 (2 Buy) (2 and))(2 Accorsi)))))))
                (2 .)))"""

    p = pickle.load(open('./assets/stoi.pkl', 'rb'))
    t = SentimentTree(tree_raw, p, 'cpu')
    print(t.root.device)
    print(t.root.left.leaf)
    print(t.root.right.right)
    print(t.get_labels())
