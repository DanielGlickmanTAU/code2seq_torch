import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a classification model')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file path',
                        required=True, type=str)

    parser.add_argument('--repeat', dest='repeat',
                        help='Repeat how many random seeds', default=1,
                        type=int)
    parser.add_argument(
        '--mark_done',
        dest='mark_done',
        action='store_true',
        help='mark yaml as yaml_done after a job has finished',
    )

    parser.add_argument('--max_examples', type=int, default=0,
                        help='limit to dataset size, useful for debugging.')
    parser.add_argument('--atom_set', type=int)
    parser.add_argument('--num_rows', type=int)
    parser.add_argument('--words_per_row', type=int)
    parser.add_argument('opts', help='See graphgym/config.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()
