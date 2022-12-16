import argparse
from unittest import TestCase
import torch

from experiments.pfedhn.trainer import train
from experiments.utils import set_logger, set_seed, get_device, str2bool


class PFedHnTest(TestCase):
    def test_basic_train(self):
        parser = argparse.ArgumentParser(
            description="Federated Hypernetwork with Lookahead experiment"
        )

        #############################
        #       Dataset Args        #
        #############################

        parser.add_argument(
            "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'],
            help="dir path for MNIST dataset"
        )
        parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
        parser.add_argument("--num-nodes", type=int, default=10, help="number of simulated nodes")

        ##################################
        #       Optimization args        #
        ##################################

        parser.add_argument("--num-steps", type=int, default=15)
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

        #############################
        #       General args        #
        #############################
        parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
        parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
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

        val_best, test_best = train(
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
            run=None,
            hyper_batch_size=args.hyper_batch_size)
        print(val_best, test_best)
        self.assertAlmostEqual(val_best, 0.24422442244224424, delta=0.01)
        self.assertAlmostEqual(test_best, 0.2342436974789916, delta=0.01)

    def test_batched_train(self):
        parser = argparse.ArgumentParser(
            description="Federated Hypernetwork with Lookahead experiment"
        )

        #############################
        #       Dataset Args        #
        #############################

        parser.add_argument(
            "--data-name", type=str, default="cifar10", choices=['cifar10', 'cifar100'],
            help="dir path for MNIST dataset"
        )
        parser.add_argument("--data-path", type=str, default="data", help="dir path for MNIST dataset")
        parser.add_argument("--num-nodes", type=int, default=10, help="number of simulated nodes")

        ##################################
        #       Optimization args        #
        ##################################

        parser.add_argument("--num-steps", type=int, default=15)
        parser.add_argument("--optim", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--inner-steps", type=int, default=50, help="number of inner steps")

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

        #############################
        #       General args        #
        #############################
        parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
        parser.add_argument("--eval-every", type=int, default=5, help="eval every X selected epochs")
        parser.add_argument("--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file")
        parser.add_argument("--seed", type=int, default=42, help="seed value")

        # ************* This changes in this test********************************
        parser.add_argument("--hyper-batch-size", type=int, default=4,
                            help="how much model HN gradients to accumulate before update")

        args = parser.parse_args()
        assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

        set_logger()
        set_seed(args.seed)

        device = get_device(gpus=args.gpu)

        if args.data_name == 'cifar10':
            args.classes_per_node = 2
        else:
            args.classes_per_node = 10

        val_best, test_best = train(
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
            run=None,
            hyper_batch_size=args.hyper_batch_size)
        print(val_best, test_best)
        self.assertAlmostEqual(val_best, 0.24422442244224424, delta=0.01)
        self.assertAlmostEqual(test_best, 0.2342436974789916, delta=0.01)
