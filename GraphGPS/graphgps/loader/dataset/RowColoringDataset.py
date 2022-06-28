import torch
from torch_geometric.data import InMemoryDataset

import graph_words.word_graphs
from graph_words.word_graphs import WordsCombinationGraphDataset

from torch_geometric.graphgym.config import cfg



class RowColoringDataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        num_samples = 10_000
        if 'max_examples' in cfg and cfg['max_examples']:
            num_samples = cfg['max_examples']
        ds = WordsCombinationGraphDataset(color_mode='rows',
                                          word_graphs=graph_words.word_graphs.get_atom_set(6),
                                          num_samples=num_samples,
                                          num_colors=20,
                                          only_color=True,
                                          unique_colors_per_example=True,
                                          words_per_row=4
                                          )
        self.data, self.slices = self.collate(ds.dataset)
        # in WOrdCombinationGraphDataset x gets values from 1 to num_colors(because of some issue with drawing the graph colors).. but here we want it to get values from 0 to num_colors-1
        self.data.x = self.data.x - self.data.x.min()
        self.data.x = self.data.x.unsqueeze(1)

    def get_idx_split(self):
        indexes = torch.arange(len(self))
        train_samples = int(len(indexes) * 0.8)
        val_samples = int(len(indexes) * 0.1)
        return {
            'train': indexes[:train_samples],
            'valid': indexes[train_samples:train_samples + val_samples],
            'test': indexes[train_samples + val_samples:]
        }
