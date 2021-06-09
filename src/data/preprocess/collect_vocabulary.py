import json
import pickle
from argparse import ArgumentParser
from collections import Counter
from os.path import dirname, join

from dpu_utils.codeutils import split_identifier_into_parts
from tqdm import tqdm


PRINT_MOST_COMMON = 10


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-d", "--data", required=True, help="Path to data corpus")
    return arg_parser


def collect_vocabulary(data_path: str):
    token_counter: Counter[str] = Counter()
    with open(data_path, "r") as data_file:
        for line in tqdm(data_file):
            raw_graph = json.loads(line)
            token_counter.update(sum([split_identifier_into_parts(t) for t in raw_graph["nodes"]], []))
    output_dir = dirname(data_path)
    with open(join(output_dir, "vocabulary.pkl"), "wb") as vocab_file:
        pickle.dump(token_counter, vocab_file)
    print(
        f"Found {len(token_counter)} tokens, "
        f"top {PRINT_MOST_COMMON} tokens: {' | '.join([t for t, _ in token_counter.most_common(PRINT_MOST_COMMON)])}"
    )


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    collect_vocabulary(__args.data)
