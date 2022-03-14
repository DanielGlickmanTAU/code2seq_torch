import sys

from code2seq.utils import compute
import comet_ml
from pytorch_lightning.loggers import CometLogger


def start_exp(exp_name, args, model):
    comet_logger = CometLogger(
        project_name=exp_name,
        api_key='FvAd5fm5rJLIj6TtmfGHUJm4b',
        workspace="danielglickmantau",
        offline=args.offline,
        save_dir=None if not args.offline else './comet_exp'
    )

    # need this for starting the experiment
    exp = comet_logger.experiment

    exp.set_model_graph(model)
    num_params = num_model_params(model)
    print(f'#Params: {num_params}')
    print(' '.join(sys.argv))

    grouped_hparams = create_hparam_id(args)

    exp.log_parameters(args)
    exp.log_parameters({'k_params': num_params / 1000, 'starting_learning_rate': str(args['learning_rate'])})
    exp.log_other('hparams_id', grouped_hparams)
    return exp


def create_hparam_id(args):
    non_relevant_hparams = ['filename', 'num_workers', 'offline', 'seed', 'exp_name', 'device']
    hparams_ = [f'{k}:{v}' for k, v in vars(args).items() if k not in non_relevant_hparams]
    hparams_ = sorted(hparams_)
    grouped_hparams = '_'.join(hparams_)
    return grouped_hparams


def num_model_params(model):
    return sum(p.numel() for p in model.parameters())
