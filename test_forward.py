# %%
import rntn
from sentiment_tensor import SentimentTree
import pickle
import torch

# %%
p = pickle.load(open('./assets/stoi.pkl', 'rb'))

lexis_size = len(p)

tree_raw = """(3 (2 It) (4 (4 (2 's)
(4 (3 (2 a) (4 (3 lovely)(2 film)))
(3 (2 with) (4 (3 (3 lovely)(2 performances))
(2 (2 by)(2 (2 (2 Buy) (2 and))(2 Accorsi)))))))
(2 .)))"""

# %%
t = SentimentTree(tree_raw, "./assets/stoi.pkl")
net = rntn.RNTensorN(lexis_size)

# %%
logits = net(t)
lb = t.get_labels()
ground_truth = torch.tensor([lb])[0]
loss_result = net.get_loss(logits, ground_truth)

# %%
