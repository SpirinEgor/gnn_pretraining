import os
import random
from argparse import ArgumentParser
from typing import Optional, Iterable

from dpu_utils.codeutils import split_identifier_into_parts

from src.data.preprocess.git_data_preparation import (
    GitProjectExtractor,
    Example,
)
from src.data.preprocess.graph_extraction import process_data

DELIMITER_EXAMPLE = "␢"
DELIMITER_FILENAME = "₣"
DEFAULT_PROJECT_NAME = "DEFAULT_PROJECT"


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "-f",
        "--data",
        type=str,
        required=True,
        help='Path to the "dataset/v3" directory or dev file with standard delimiters',
    )
    arg_parser.add_argument(
        "-t",
        "--dest",
        type=str,
        required=True,
        help="Path to directory where graphs will be stored",
    )
    arg_parser.add_argument(
        "--val_part",
        type=float,
        default=0.2,
        help="Part of the dataset which will be used for validation",
    )
    arg_parser.add_argument(
        "--test_part",
        type=float,
        default=0.2,
        help="Part of the dataset which will be used for testing",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for projects and examples shuffle",
    )
    arg_parser.add_argument(
        "--vocabulary", action="store_true", help="if passed then collect vocabulary from train holdout"
    )

    return arg_parser


def __vocabulary_for_func(graph: dict) -> Iterable[str]:
    return (split_identifier_into_parts(t) for t in graph["nodes"])


def preprocess(
    data_path: str,
    dest_path: str,
    random_seed: int,
    val_part: Optional[float] = None,
    test_part: Optional[float] = None,
    need_vocabulary: bool = False,
):
    if need_vocabulary:
        vocabulary_for = __vocabulary_for_func
    else:
        vocabulary_for = None

    if os.path.isfile(data_path):
        # Dev case
        with open(data_path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        text_examples = [example for example in text.split(DELIMITER_EXAMPLE) if example]
        examples = []
        for text_example in text_examples:
            file_name, source_code = map(str.strip, text_example.split(DELIMITER_FILENAME))
            examples.append(
                Example(
                    language="Python", project_name=DEFAULT_PROJECT_NAME, file_name=file_name, source_code=source_code
                )
            )
        rng = random.Random(random_seed)
        rng.shuffle(examples)

        if test_part is None:
            test_part = 0.0
        if val_part is None:
            val_part = 0.0
        test_size = int(len(examples) * test_part)
        val_size = int(len(examples) * val_part)

        test_examples, val_examples, train_examples = (
            examples[:test_size],
            examples[test_size : test_size + val_size],
            examples[test_size + val_size :],
        )
        process_data(test_examples, "test", dest_path)
        process_data(val_examples, "val", dest_path)
        process_data(train_examples, "train", dest_path, vocabulary_func=vocabulary_for)
    else:
        data_extractor = GitProjectExtractor(data_path, random_seed, val_part, test_part)

        if val_part:
            process_data(data_extractor, "val", dest_path)
        if test_part:
            process_data(data_extractor, "test", dest_path)
        process_data(data_extractor, "train", dest_path, vocabulary_func=vocabulary_for)


if __name__ == "__main__":
    parser = configure_arg_parser()
    args = parser.parse_args()

    preprocess(
        args.data,
        args.dest,
        random_seed=args.seed,
        val_part=args.val_part,
        test_part=args.test_part,
        need_vocabulary=args.vocabulary,
    )
