"""
Data loader for training
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GraphDataset(Dataset):
    def __init__(self, features, labels, edge_indices, indices=None):
        self.features = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()

        # Handle edge_indices - can be either numpy arrays or tensors
        self.edge_indices = []
        for ei in edge_indices:
            if isinstance(ei, np.ndarray):
                self.edge_indices.append(torch.from_numpy(ei).long())
            else:
                self.edge_indices.append(ei.long())

        self.indices = indices if indices is not None else np.arange(len(labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        node_idx = self.indices[idx]
        x = self.features[node_idx]
        y = self.labels[node_idx]
        # Return only node features and labels, not edge_indices (shared across batch)
        return x, y


def create_data_loaders(features, labels, edge_indices, train_idx, val_idx, test_idx, batch_size=1024):
    train_dataset = GraphDataset(features, labels, edge_indices, train_idx)
    val_dataset = GraphDataset(features, labels, edge_indices, val_idx)
    test_dataset = GraphDataset(features, labels, edge_indices, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
