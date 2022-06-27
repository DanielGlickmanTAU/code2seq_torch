import torch
from torch_geometric.data import InMemoryDataset

import graph_words.word_graphs
from graph_words.word_graphs import WordsCombinationGraphDataset


class RowColoringDataset(InMemoryDataset):
    def __init__(self):
        ds = WordsCombinationGraphDataset(color_mode='rows',
                                          word_graphs=graph_words.word_graphs.get_atom_set(6),
                                          num_samples=1000,
                                          num_colors=2,
                                          only_color=True,
                                          unique_colors_per_example=True,
                                          words_per_sample=4
                                          )
        super().__init__()
        self.data, self.slices = self.collate(ds.dataset)

    def get_idx_split(self):
        indexes = torch.arange(len(self))
        train_samples = int(len(indexes) * 0.8)
        val_samples = int(len(indexes) * 0.1)
        return {
            'train': indexes[:train_samples],
            'valid': indexes[train_samples:train_samples + val_samples],
            'test': indexes[train_samples + val_samples:]
        }
