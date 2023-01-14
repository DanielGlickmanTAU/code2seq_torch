from unittest import TestCase

from experiments.pfedhn.misc.args import get_args
from experiments.pfedhn.trainer import train
from experiments.utils import set_logger, set_seed, get_device


class PFedHnTest(TestCase):
    def test_basic_train(self):
        args = get_args()
        args.hyper_batch_size = 1
        args.num_steps = 15
        args.num_nodes = 10
        args.eval_every = 5
        set_logger()
        set_seed(args.seed)

        device = get_device(gpus=args.gpu)

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
            hyper_batch_size=args.hyper_batch_size,
            args=args)
        print(val_best, test_best)
        self.assertAlmostEqual(val_best, 0.26327632763276326, delta=0.01)
        self.assertAlmostEqual(test_best, 0.2560024009603842, delta=0.01)

    def test_batched_train(self):
        args = get_args()
        args.hyper_batch_size = 4
        args.num_steps = 15
        args.num_nodes = 10
        args.eval_every = 5

        set_logger()
        set_seed(args.seed)

        device = get_device(gpus=args.gpu)

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
            hyper_batch_size=args.hyper_batch_size,
            args=args)
        print(val_best, test_best)

        self.assertAlmostEqual(val_best, 0.4273927392739274, delta=0.01)
        self.assertAlmostEqual(test_best, 0.4272208883553421, delta=0.01)
