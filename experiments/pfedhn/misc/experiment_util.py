from custom.info import get_wandb
from graphgps.utils import cfg_to_dict


def init_wandb(args):
    try:
        import wandb
    except:
        raise ImportError('WandB is not installed.')
    # if cfg.wandb.name == '':
    #     wandb_name = make_wandb_name(cfg)
    # else:
    #     wandb_name = cfg.wandb.name
    try:
        run = wandb.init(entity='daniel-ai', project='federated_attn',
                         name='')
    except Exception:
        wandb.login(key=get_wandb())
        run = wandb.init(entity='daniel-ai', project='federated_attn',
                         name='')
    run.config.update(args)
    return run


def parse_results_to_wandb_log_format(results):
    new_res = {}
    for key, value in results.items():
        value = value[-1]
        prefixes = ['best', 'train', 'val', 'test']
        for prefix in prefixes:
            if prefix in key:
                prefix = f'{prefix}/{key}'
                break
        new_res[key] = value

    return new_res
