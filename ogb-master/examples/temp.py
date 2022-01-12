from ogb.graphproppred import GraphPropPredDataset, PygGraphPropPredDataset
# import torch_sparse
# from ogb.graphproppred import PygGraphPropPredDataset

from torch_geometric.loader import DataLoader

# Download and process data at './dataset/ogbg_molhiv/'
dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
# dataset = GraphPropPredDataset(name='ogbg-molhiv')

split_idx = dataset.get_idx_split()
train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx['test']], batch_size=32, shuffle=False)
# pip install --force-reinstall --no-deps joblib==1.1.0'
#pip install --force-reinstall --no-deps joblib==1.1.0'