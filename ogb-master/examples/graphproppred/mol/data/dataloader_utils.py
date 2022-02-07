from torch_geometric.loader import DataLoader


def get_train_val_test_loaders(dataset, batch_size, num_workers, limit=0):
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
