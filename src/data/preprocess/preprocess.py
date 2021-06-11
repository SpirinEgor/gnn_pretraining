import gzip
import os
import json
import logging
import pickle
import random
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count, Queue, Manager
from typing import Dict, Optional, Union, List

from dpu_utils.codeutils import split_identifier_into_parts
from tqdm.auto import tqdm

from src.data.preprocess.collect_vocabulary import PRINT_MOST_COMMON
from src.data.preprocess.git_data_preparation import (
    GitProjectExtractor,
    Example,
)
from src.data.preprocess.typilus.graphgenerator import AstGraphGenerator
from src.data.preprocess.typilus.type_lattice_generator import (
    TypeLatticeGenerator,
)

DELIMITER_EXAMPLE = "␢"
DELIMITER_FILENAME = "₣"
DEFAULT_PROJECT_NAME = "DEFAULT_PROJECT"
TYPE_LATTICE_CONFIG = os.path.join(os.getcwd(), os.path.dirname(__file__), "typilus/type_lattice_config.json")

USE_CPU = cpu_count()

LOG_FILENAME = "log.txt"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, filemode="w")
logger = logging.getLogger(__name__)


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


@dataclass
class QueueMessage:
    graph: Optional[Dict] = None
    error: Optional[str] = None
    is_finished: bool = False


def handle_queue_message(queue: Queue, output_file: str):
    with gzip.open(output_file, "wb", compresslevel=1) as gzip_file:
        while True:  # Should be ok since function implemented for async usage
            message: QueueMessage = queue.get()
            if message.is_finished:
                break
            if message.graph is None:
                logger.error(message.error)
                continue
            gzip_file.write((json.dumps(message.graph) + "\n").encode("utf-8"))


def extract_graph(example: Example) -> Dict:
    type_lattice = TypeLatticeGenerator(TYPE_LATTICE_CONFIG)
    visitor = AstGraphGenerator(example.source_code, type_lattice)
    graph = visitor.build()
    graph["file_name"] = example.file_name
    graph["project_name"] = example.project_name
    return graph


def extract_graph_parallel(example: Example, queue: Queue, vocabulary: bool = False) -> Optional[Counter]:
    try:
        graph = extract_graph(example)
    except Exception as e:
        error = f"Can't generate graph from {example.file_name}, exception: {e}"
        queue.put(QueueMessage(None, error))
        return None
    if graph is None or len(graph["supernodes"]) == 0:
        error = f"Graph without supernodes in {example.file_name}"
        queue.put(QueueMessage(None, error))
        return None

    queue.put(QueueMessage(graph, None))
    if vocabulary:
        return Counter(
            sum(
                [split_identifier_into_parts(t) for t in graph["nodes"]],
                [],
            )
        )
    return None


def process_holdout(
    data: Union[GitProjectExtractor, List[Example]],
    holdout: str,
    dest_path: str,
    need_vocabulary: bool = False,
) -> None:
    os.makedirs(dest_path, exist_ok=True)
    output_file = os.path.join(dest_path, f"graphs_{holdout}.jsonl.gz")

    examples = data.get_examples(holdout) if isinstance(data, GitProjectExtractor) else data

    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)

        pool.apply_async(handle_queue_message, (message_queue, output_file))

        process_func = partial(extract_graph_parallel, queue=message_queue, vocabulary=need_vocabulary)
        counters: List = pool.map(process_func, tqdm(examples, desc=f"Processing graphs from {holdout}"))

        message_queue.put(QueueMessage(None, None, True))

        pool.close()
        pool.join()

    if need_vocabulary:
        assert counters is not None, "no counters collected during graphs preprocessing"
        token_counter: Counter[str] = sum(filter(lambda c: c is not None, counters), Counter())
        print(
            f"Found {len(token_counter)} tokens, "
            f"top {PRINT_MOST_COMMON} tokens: "
            f"{' | '.join([t for t, _ in token_counter.most_common(PRINT_MOST_COMMON)])}"
        )
        with open(os.path.join(dest_path, "vocabulary.pkl"), "wb") as vocab_file:
            pickle.dump(token_counter, vocab_file)


def preprocess(
    data_path: str,
    dest_path: str,
    random_seed: int,
    val_part: Optional[float] = None,
    test_part: Optional[float] = None,
    need_vocabulary: bool = False,
):

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
        process_holdout(test_examples, "test", dest_path)
        process_holdout(val_examples, "val", dest_path)
        process_holdout(train_examples, "train", dest_path, need_vocabulary)
    else:
        data_extractor = GitProjectExtractor(data_path, random_seed, val_part, test_part)

        if val_part:
            process_holdout(data_extractor, "val", dest_path)
        if test_part:
            process_holdout(data_extractor, "test", dest_path)
        process_holdout(data_extractor, "train", dest_path, need_vocabulary)


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
