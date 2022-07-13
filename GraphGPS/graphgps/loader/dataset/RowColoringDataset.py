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
        atom_set = 6
        if 'atom_set' in cfg and cfg['atom_set']:
            atom_set = cfg['atom_set']

        if 'num_rows' in cfg and cfg['num_rows'] > 0:
            words_per_row = cfg['words_per_row']
            num_rows = cfg['num_rows']
        else:
            num_rows = words_per_row = 4

        only_color = cfg.dataset.only_color
        ds = WordsCombinationGraphDataset(color_mode='rows',
                                          word_graphs=graph_words.word_graphs.get_atom_set(atom_set),
                                          num_samples=num_samples,
                                          num_colors=cfg.dataset.node_encoder_num_types,
                                          only_color=only_color,
                                          unique_colors_per_example=True,
                                          # if tagging by shape, draw unique shape per instace, to get reasonable probability.
                                          unique_atoms_per_example=not only_color,
                                          words_per_row=words_per_row,
                                          num_rows=num_rows,
                                          num_unique_atoms=cfg['num_unique_atoms'],
                                          num_unique_colors=cfg['num_unique_colors']
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
