# %%

import collections
import torch
import torch.nn as nn
# from torch.linalg import norm
# %%


class RNTensorN(nn.Module):
    def __init__(self, lexis_size, embed_size=100, num_classes=5):
        super(RNTensorN, self).__init__()
        self.embedding = nn.Embedding(int(lexis_size), embed_size)
        self.V = nn.Parameter(torch.randn(2 * embed_size, 2 * embed_size, embed_size))
        self.W = nn.Linear(2 * embed_size, embed_size, bias=True)
        self.Ws = nn.Linear(embed_size, num_classes, bias=True)
        self.activation = nn.Tanh()
        self.classifier = nn.Softmax(dim=1)
        self.embed_size = embed_size

    def traverse(self, node):
        full_nodes_rep = collections.OrderedDict()

        if node.leaf:
            node.h = self.embedding(torch.LongTensor([node[1].item()]))
        else:
            left_node_dict = self.traverse(node.left)
            left_tensor = left_node_dict[node.left]
            right_node_dict = self.traverse(node.right)
            right_tensor = right_node_dict[node.right]

            full_nodes_rep.update(left_node_dict)
            full_nodes_rep.update(right_node_dict)

            concat = torch.cat((left_tensor, right_tensor), 1)
            # main neural tensor action
            main_rntn_tmp = torch.mm(concat, self.V.view(2 * self.embed_size, -1))
            main_rntn_ret = torch.mm(main_rntn_tmp.view(self.embed_size, 2 * self.embed_size), concat.transpose(0, 1))

            # neural tensor output + standard layer output
            composed = main_rntn_ret.transpose(0, 1) + self.W(concat)
            node.h = self.activation(composed)

        full_nodes_rep[node] = node.h
        return full_nodes_rep

    def forward(self, root):
        nodes_ret = self.traverse(root)
        self.nodes_rep_matrix = torch.cat(list(nodes_ret.values()))

        self.tree_logits = self.classifier(self.Ws(self.nodes_rep_matrix))
        return self.tree_logits

    def get_loss(self, prediction, target):
        loss_f = nn.CrossEntropyLoss(reduction='mean')
        loss_raw = loss_f(prediction, target)

        """Not needed to calculate L2 penalty since
        weight decay computation goes into the optimizer."""

        """
        l2_embed = norm(self.embedding.weight, ord=2)
        l2_W = norm(self.W.weight, ord=2)
        l2_V = norm(self.V.view(2 * self.embed_size, -1), ord=2)
        l2_Ws = norm(self.Ws.weight, ord=2)
        l2_terms = (l2_W + l2_V + l2_Ws + l2_embed)
        """
        tree_loss = loss_raw  # + self.l2_factor * l2_terms

        mle = torch.argmax(prediction, dim=1)
        tree_accuracy = torch.sum((mle == target)) / len(target)

        return tree_loss, tree_accuracy
