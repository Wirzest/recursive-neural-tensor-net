# -*- coding: utf-8 -*-
# author: Victor H. Wirz
# discipline: UNIRIO-tin0145
# prof.: Pedro Moura

from nltk.tree import ParentedTree

_UNK = '<unk>'


class SentimentTree():
    r"""
    Hierarchical structure used to read and represent the
    Stanford Treebank.

    It uses the nltk implementation of fromstring() to catch the
    representation of syntax trees and morphological trees. But
    the ending object is not an instance of any nklt package classes.
    """
    def __init__(self, tree_raw, stoi):
        nltk_tree = ParentedTree.fromstring(tree_raw)

        for leaf_idx in nltk_tree.treepositions('leaves'):
            if nltk_tree[leaf_idx] in stoi:
                nltk_tree[leaf_idx] = stoi[nltk_tree[leaf_idx]]
            else:
                nltk_tree[leaf_idx] = stoi[_UNK]

        self.root = self.parse(nltk_tree)
        self.labels = self._get_labels_(self.root)

    def parse(self, nltk_tree):
        if nltk_tree.height() == 2:
            return Node(nltk_tree.label(), True, word=nltk_tree[0])

        node = Node(nltk_tree.label(), False)
        node.set_left(self.parse(nltk_tree[0]))
        node.set_right(self.parse(nltk_tree[1]))
        return node

    def _get_labels_(self, node):
        if node is None:
            return []

        return self._get_labels_(node.left) + self._get_labels_(node.right) + [int(node.label)]

    def get_labels(self):
        return self.labels


class Node():
    def __init__(self, label, is_leaf, right=None, left=None, word=None):
        """
        label: one-hot vector
        h: activations of node after composition ops
        """
        self.label = label
        self.word = word
        self.parent = None
        self.left = left
        self.right = right
        self.leaf = is_leaf
        self.h = None
        self.logits = None  # optionally assign these for alt implementation
        self.loss = None

    def set_left(self, left):
        self.left = left

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def set_right(self, right):
        self.right = right

    def get_word(self):
        return self.word

    def is_leaf(self):
        return self.leaf


if __name__ == 'main':
    import pickle

    tree_raw = """(3 (2 It) (4 (4 (2 's)
                (4 (3 (2 a) (4 (3 lovely)(2 film)))
                (3 (2 with) (4 (3 (3 lovely)(2 performances))
                (2 (2 by)(2 (2 (2 Buy) (2 and))(2 Accorsi)))))))
                (2 .)))"""

    p = pickle.load(open('./assets/stoi.pkl', 'rb'))
    t = SentimentTree(tree_raw, p)
