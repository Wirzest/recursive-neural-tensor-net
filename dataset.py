import torch
import numpy as np
# from sentiment_tree import SentimentTree
from torch.utils.data import Dataset


class SSTDataset(Dataset):
    def __init__(self, file_dir, stoi, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subdirectory images
            transform (callable, optional): Optional transform to be applied
                on the image.
            target_transform (callable, optional): Optional transform to be applied
                on the target.
        """
        self.file_dir = file_dir
        self.stoi = stoi

        trees = []
        with open(file_dir) as f:
            for line in f:
                trees.append(line)  # raw tree strings

        self.trees = np.array(trees)
        self.transform = transform
        trees *= 0

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the datum to be returned.
        Returns:
            A dict containing the image with required
            transformations along its label in tensor format.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tree = self.trees[idx]

        if self.transform:
            tree = self.transform(tree)

        return tree
