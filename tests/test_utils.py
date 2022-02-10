from typing import Union, List

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


def as_pyg_batch(dataset: Union[Dataset, List[Data]], batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    return list(loader)[0]
