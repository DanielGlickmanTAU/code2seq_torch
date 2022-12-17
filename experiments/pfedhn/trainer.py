import argparse
import json
import logging
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

from experiments.pfedhn.misc.experiment_util import init_wandb, parse_results_to_wandb_log_format

import numpy as np
import torch
import torch.utils.data
from tqdm import trange

from experiments.pfedhn.models import CNNHyper, CNNTarget
from experiments.pfedhn.node import BaseNodes
from experiments.utils import get_device, set_logger, set_seed, str2bool


def eval_model(nodes, num_nodes, hnet, net, criteria, device, split):
    curr_results = evaluate(nodes, num_nodes, hnet, net, criteria, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_loss, avg_acc, all_acc


@torch.no_grad()
def evaluate(nodes: BaseNodes, num_nodes, hnet, net, criteria, device, split='test'):
    def get_loader_for_node(node_id):
        if split == 'test':
            return nodes.test_loaders[node_id]
        elif split == 'val':
            return nodes.val_loaders[node_id]
        else:
            return nodes.train_loaders[node_id]

    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    ids_tensor = torch.tensor(list(range(num_nodes)), dtype=torch.long, device=device).view(-1)
    emds = hnet.embeddings(ids_tensor)
    weights_batched = hnet(emds)
    for node_id in range(num_nodes):  # iterating over nodes

        running_loss, running_correct, running_samples = 0., 0., 0.

        curr_data = get_loader_for_node(node_id)
        weights = OrderedDict({k: tensor[node_id] for k, tensor in weights_batched.items()})
        net.load_state_dict(weights)
        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            pred = net(img)

            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)
        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results


def train(data_name: str, data_path: str, classes_per_node: int, num_nodes: int,
          steps: int, inner_steps: int, optim: str, lr: float, inner_lr: float,
          embed_lr: float, wd: float, inner_wd: float, embed_dim: int, hyper_hid: int,
          n_hidden: int, n_kernels: int, bs: int, device, eval_every: int, save_path: Path,
          seed: int, run, hyper_batch_size, embedding_type='') -> None:
    ###############################
    # init nodes, hnet, local net #
    ###############################
    nodes = BaseNodes(data_name, data_path, num_nodes, classes_per_node=classes_per_node,
                      batch_size=bs)

    embed_dim = embed_dim
    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1 + num_nodes / 4)

    if data_name == "cifar10":
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden, n_kernels=n_kernels,
                        embedding_type=embedding_type)
        net = CNNTarget(n_kernels=n_kernels)
    elif data_name == "cifar100":
        hnet = CNNHyper(num_nodes, embed_dim, hidden_dim=hyper_hid,
                        n_hidden=n_hidden, n_kernels=n_kernels, out_dim=100)
        net = CNNTarget(n_kernels=n_kernels, out_dim=100)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    hnet = hnet.to(device)
    net = net.to(device)

    ##################
    # init optimizer #
    ##################
    embed_lr = embed_lr if embed_lr is not None else lr
    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ], lr=lr, momentum=0.9, weight_decay=wd
        ),
        'adam': torch.optim.Adam(params=hnet.parameters(), lr=lr)
    }
    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = defaultdict(list)
    for step in step_iter:
        # select client at random
        nodes_id = random.sample(range(num_nodes), hyper_batch_size)
        ids_tensor = torch.tensor([nodes_id], dtype=torch.long, device=device).view(-1)
        emds = hnet.embeddings(ids_tensor)
        hnet_grads, train_acc = compute_hn_grads(criteria, device, hnet, inner_lr, inner_steps, inner_wd,
                                                 net,
                                                 emds,
                                                 nodes_id, nodes,
                                                 optimizer)

        # update hnet weights
        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
        optimizer.step()

        # step_iter.set_description(
        #     f"Step: {step + 1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
        # )
        results['train_acc'].append(train_acc)
        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device,
                                                                  split="test")
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)
            if run:
                run.log(parse_results_to_wandb_log_format(results), step=step)

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
        step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device,
                                                              split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)
        if run:
            run.log(parse_results_to_wandb_log_format(results), step=step)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(str(save_path / f"results_{inner_steps}_inner_steps_seed_{seed}.json"), "w") as file:
        json.dump(results, file, indent=4)
    return best_acc, test_best_based_on_step


def compute_hn_grads(criteria, device, hnet, inner_lr, inner_steps, inner_wd, net, emd, nodes_id, nodes, optimizer):
    hnet.train()
    # produce & load local network weights

    weights = hnet(emd)

    hnet_grads, train_acc = compute_grads_of_client_net(criteria, device, hnet, inner_lr, inner_steps, inner_wd, net,
                                                        nodes_id,
                                                        nodes, optimizer, weights)
    return hnet_grads, train_acc


def compute_grads_of_client_net(criteria, device, hnet, inner_lr, inner_steps, inner_wd, net, nodes_id, nodes,
                                optimizer,
                                weights_batched):
    acc, loss = 0., 0.
    deltas = OrderedDict({k: torch.zeros_like(tensor) for k, tensor in weights_batched.items()})
    for batch_index in range(len(nodes_id)):
        node_id = nodes_id[batch_index]
        weights = OrderedDict({k: tensor[batch_index] for k, tensor in weights_batched.items()})
        net.load_state_dict(weights)
        # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})
        # NOTE: evaluation on sent model
        prvs_acc, prvs_loss = eval_client_model(criteria, device, net, node_id, nodes)
        final_state = get_trained_network_state(criteria, device, inner_lr, inner_steps, inner_wd, net, node_id, nodes,
                                                optimizer)
        # calculating delta theta
        # delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})
        for k in weights.keys():
            deltas[k][batch_index] = inner_state[k] - final_state[k]
        loss += prvs_loss
        acc += prvs_acc
    # calculating phi gradient
    # before here I need to batch the weights and deltas...
    hnet_grads = torch.autograd.grad(
        list(weights_batched.values()), hnet.parameters(), grad_outputs=list(deltas.values())
    )
    acc = acc / len(nodes_id)
    loss = loss / len(nodes_id)
    return hnet_grads, acc


def get_trained_network_state(criteria, device, inner_lr, inner_steps, inner_wd, net, node_id, nodes, optimizer):
    # init inner optimizer
    inner_optim = torch.optim.SGD(
        net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
    )
    # inner updates -> obtaining theta_tilda
    for i in range(inner_steps):
        net.train()
        inner_optim.zero_grad()
        optimizer.zero_grad()

        batch = next(iter(nodes.train_loaders[node_id]))
        img, label = tuple(t.to(device) for t in batch)

        pred = net(img)

        loss = criteria(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 50)

        inner_optim.step()
    optimizer.zero_grad()
    final_state = net.state_dict()
    return final_state


def eval_client_model(criteria, device, net, node_id, nodes):
    with torch.no_grad():
        net.eval()
        batch = next(iter(nodes.test_loaders[node_id]))
        img, label = tuple(t.to(device) for t in batch)
        pred = net(img)
        prvs_loss = criteria(pred, label)
        prvs_acc = pred.argmax(1).eq(label).sum().item() / len(label)
        net.train()
    print(prvs_acc, prvs_loss)
    return prvs_acc, prvs_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetwork with Lookahead experiment"
    )

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'], help="dir path for MNIST dataset"
    )
    parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
    parser.add_argument("--num-nodes", type=int, default=50, help="number of simulated nodes")

    ##################################
    #       Optimization args        #
    ##################################

    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")
    parser.add_argument("--hyper-batch-size", type=int, default=1,
                        help="how much model HN gradients to accumulate before update")

    ################################
    #       Model Prop args        #
    ################################
    parser.add_argument("--n-hidden", type=int, default=3, help="num. hidden layers")
    parser.add_argument("--inner-lr", type=float, default=5e-3, help="learning rate for inner optimizer")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--inner-wd", type=float, default=5e-5, help="inner weight decay")
    parser.add_argument("--embed-dim", type=int, default=-1, help="embedding dim")
    parser.add_argument("--embed-lr", type=float, default=None, help="embedding learning rate")
    parser.add_argument("--hyper-hid", type=int, default=100, help="hypernet hidden dim")
    parser.add_argument("--spec-norm", type=str2bool, default=False, help="hypernet hidden dim")
    parser.add_argument("--nkernels", type=int, default=16, help="number of kernels for cnn model")
    parser.add_argument("--embedding_type", type=str, default='', help="mix embeddings with attention")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
    parser.add_argument("--eval-every", type=int, default=30, help="eval every X selected epochs")
    parser.add_argument("--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus=args.gpu)

    if args.data_name == 'cifar10':
        args.classes_per_node = 2
    else:
        args.classes_per_node = 10

    run = init_wandb(args)

    train(
        data_name=args.data_name,
        data_path=args.data_path,
        classes_per_node=args.classes_per_node,
        num_nodes=args.num_nodes,
        steps=args.num_steps,
        inner_steps=args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        save_path=args.save_path,
        seed=args.seed,
        run=run,
        hyper_batch_size=args.hyper_batch_size,
        embedding_type=args.embedding_type,
    )
