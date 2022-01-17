import args_parse
from code2seq.utils import compute
from gnn import GNN

compute.get_torch()
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torchvision import transforms

from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import os

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

### importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
### for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq

multicls_criterion = torch.nn.CrossEntropyLoss()


def train(model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:, i])

            loss = loss / len(pred_list)

            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))


def eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code2 data with Pytorch Geometrics')
    args_parse.add_args(parser)
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--max_seq_len', type=int, default=10)
    parser.add_argument('--random_split', action='store_true')
    parser.add_argument('--dataset', type=str, default="ogbg-code2",
                        help='dataset name (default: ogbg-code2)')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(args.max_seq_len,
                                                              np.sum(seq_len_list <= args.max_seq_len) / len(
                                                                  seq_len_list)))

    split_idx = dataset.get_idx_split()

    if args.random_split:
        print('Using random split')
        perm = torch.randperm(len(dataset))
        num_train, num_valid, num_test = len(split_idx['train']), len(split_idx['valid']), len(split_idx['test'])
        split_idx['train'] = perm[:num_train]
        split_idx['valid'] = perm[num_train:num_train + num_valid]
        split_idx['test'] = perm[num_train + num_valid:]

        assert (len(split_idx['train']) == num_train)
        assert (len(split_idx['valid']) == num_valid)
        assert (len(split_idx['test']) == num_test)

    ### building vocabulary for sequence predition. Only use training data.

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    ### set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose(
        [augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    print(nodeattributes_mapping)

    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes=len(nodetypes_mapping['type']),
                                  num_nodeattributes=len(nodeattributes_mapping['attr']), max_depth=20)

    if args.gnn == 'gin':
        model = GNN(args, num_tasks=len(vocab2idx), node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(args, num_tasks=len(vocab2idx), node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gin', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(args, num_tasks=len(vocab2idx), node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(args, num_tasks=len(vocab2idx), node_encoder=node_encoder,
                    num_layer=args.num_layer, gnn_type='gcn', emb_dim=args.emb_dim, drop_ratio=args.drop_ratio,
                    virtual_node=True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f'#Params: {sum(p.numel() for p in model.parameters())}')

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator,
                          arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        valid_perf = eval(model, device, valid_loader, evaluator,
                          arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))
        test_perf = eval(model, device, test_loader, evaluator,
                         arr_to_seq=lambda arr: decode_arr_to_seq(arr, idx2vocab))

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    print('F1')
    best_val_epoch = np.argmax(np.array(valid_curve))
    best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        result_dict = {'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch],
                       'Train': train_curve[best_val_epoch], 'BestTrain': best_train}
        torch.save(result_dict, args.filename)


if __name__ == "__main__":
    main()
