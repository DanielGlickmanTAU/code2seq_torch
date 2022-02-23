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
    exp.log_parameters(args)
    exp.log_parameters({'k_params': num_params / 1000})
    return exp


def num_model_params(model):
    return sum(p.numel() for p in model.parameters())
