import dataclasses
import functools
import gzip
import itertools
import json
import logging
import os
from collections import Counter
from multiprocessing import Queue, cpu_count, Manager, Pool
from typing import Optional, Union, Callable, Iterable

from datasets import tqdm

from src.data.preprocess.example import Example
from src.data.preprocess.git_data_preparation import GitProjectExtractor
from src.data.preprocess.typilus.graphgenerator import AstGraphGenerator
from src.data.preprocess.typilus.type_lattice_generator import TypeLatticeGenerator


LOG_FILENAME = "log.txt"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, filemode="w")
logger = logging.getLogger(__name__)
USE_CPU = cpu_count()

TYPE_LATTICE_CONFIG = os.path.join(os.getcwd(), os.path.dirname(__file__), "typilus/type_lattice_config.json")
PRINT_MOST_COMMON = 10


def process_data(
    data: Union[GitProjectExtractor, list[Example]],
    holdout: str,
    save_dir_path: str,
    vocabulary_func: Optional[Callable[[dict], Iterable[str]]] = None,
) -> None:
    os.makedirs(save_dir_path, exist_ok=True)
    output_file = os.path.join(save_dir_path, f"graphs_{holdout}.jsonl.gz")

    examples = data.get_examples(holdout) if isinstance(data, GitProjectExtractor) else data
    examples_length = data.get_num_examples(holdout) if isinstance(data, GitProjectExtractor) else len(examples)

    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)

        graphs_counter = pool.apply_async(handle_queue_message, (message_queue, output_file))

        process_func = functools.partial(extract_graph_parallel, queue=message_queue, vocabulary_func=vocabulary_func)
        counters: list = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, examples),
                desc=f"Processing graphs from {holdout}...",
                total=examples_length,
            )
        ]

        message_queue.put(QueueMessage(None, None, True))

        pool.close()
        pool.join()
        print(f"Extracted {graphs_counter.get()} graphs")

    if vocabulary_func is not None:
        assert counters is not None, "no counters collected during graphs preprocessing"
        token_counter: Counter[str] = Counter()
        for c in counters:
            if c is not None:
                token_counter.update(c)
        print(
            f"Found {len(token_counter)} tokens, "
            f"top {PRINT_MOST_COMMON} tokens: "
            f"{' | '.join([t for t, _ in token_counter.most_common(PRINT_MOST_COMMON)])}"
        )
        with open(os.path.join(save_dir_path, "counter.json"), "w") as f:
            json.dump(dict(token_counter), f)


@dataclasses.dataclass
class QueueMessage:
    graph: Optional[dict] = None
    error: Optional[str] = None
    is_finished: bool = False


def handle_queue_message(queue: Queue, output_file: str) -> int:
    graphs_counter = 0
    with gzip.open(output_file, "wb", compresslevel=1) as gzip_file:
        while True:  # Should be ok since function implemented for async usage
            message: QueueMessage = queue.get()
            if message.is_finished:
                break
            if message.graph is None:
                logger.error(message.error)
                continue
            gzip_file.write((json.dumps(message.graph) + "\n").encode("utf-8"))
            graphs_counter += 1
    return graphs_counter


def extract_graph(example: Example) -> dict:
    type_lattice = TypeLatticeGenerator(TYPE_LATTICE_CONFIG)
    visitor = AstGraphGenerator(example.source_code, type_lattice)
    graph = visitor.build()
    graph.update(dataclasses.asdict(example))
    return graph


def extract_graph_parallel(
    example: Example, queue: Queue, vocabulary_func: Optional[Callable[[dict], Iterable[str]]] = None
) -> Optional[Counter]:
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
    if vocabulary_func is not None:
        cntr = Counter(itertools.chain(vocabulary_func(graph)))
        return cntr
    return None
