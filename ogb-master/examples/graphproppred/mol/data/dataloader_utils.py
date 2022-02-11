from torch_geometric.data import Dataset
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.loader import DataLoader


def get_train_val_test_loaders(dataset: GNNBenchmarkDataset, batch_size, num_workers, limit=0):
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]][:limit] if limit else dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]][:limit] if limit else dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]][:limit] if limit else dataset[split_idx["test"]]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, valid_loader, test_loader


def pyg_get_train_val_test_loaders(name, batch_size, num_workers, transform=None, limit=0,
                                   ):
    train_dataset = GNNBenchmarkDataset(root=('./%s' % name), name=('%s' % name), split='train', transform=transform)
    val_dataset = GNNBenchmarkDataset(root=name, name=('%s' % name), split='val', transform=transform)
    test_dataset = GNNBenchmarkDataset(root=name, name=('%s' % name), split='test', transform=transform)

    train_dataset = train_dataset[: limit] if limit else train_dataset["train"]
    val_dataset = val_dataset[: limit] if limit else val_dataset["valid"]
    test_dataset = test_dataset[: limit] if limit else test_dataset["test"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, valid_loader, test_loader
