from code2seq.utils import compute
import comet_ml
from pytorch_lightning.loggers import CometLogger


def start_exp(exp_name, args, model):
    comet_logger = CometLogger(
        project_name=exp_name,
        api_key='FvAd5fm5rJLIj6TtmfGHUJm4b',
        workspace="danielglickmantau",
    )

    # need this for starting the experiment
    exp = comet_logger.experiment

    exp.set_model_graph(model)
    exp.log_parameters(args)
    return exp
