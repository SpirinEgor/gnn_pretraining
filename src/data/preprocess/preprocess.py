import gzip
import os
import json
import logging
import random
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional

from tqdm.auto import tqdm

from src.data.preprocess.git_data_preparation import GitProjectExtractor, Example
from src.data.preprocess.typilus.graphgenerator import AstGraphGenerator
from src.data.preprocess.typilus.type_lattice_generator import TypeLatticeGenerator

TYPE_LATTICE_CONFIG = os.path.join(os.getcwd(), os.path.dirname(__file__), "typilus/type_lattice_config.json")

USE_CPU = cpu_count()

LOG_FILENAME = "log.txt"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, filemode="w")
logger = logging.getLogger(__name__)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--data", type=str, required=True, help='Path to the "dataset/v3" directory')
    arg_parser.add_argument(
        "-t", "--dest", type=str, required=True, help="Path to directory where graphs will be stored"
    )
    arg_parser.add_argument(
        "--train_part", type=float, default=0.7, help="Part of the dataset which will be used for training"
    )
    arg_parser.add_argument(
        "--val_part", type=float, default=0.2, help="Part of the dataset which will be used for validation"
    )
    arg_parser.add_argument(
        "--test_part", type=float, default=0.2, help="Part of the dataset which will be used for testing"
    )
    arg_parser.add_argument("--seed", type=int, default=17, help="Random seed for projects and examples shuffle")

    return arg_parser


def extract_graphs(example: Example) -> Optional[Dict]:
    type_lattice = TypeLatticeGenerator(TYPE_LATTICE_CONFIG)
    try:
        visitor = AstGraphGenerator(example.source_code, type_lattice)
        graph = visitor.build()
    except Exception as e:
        # TODO: proper logging in multiprocessing
        logging.error(f"Can't generate graph from {example.file_name}, exception: {e}")
        return None

    graph["file_name"] = example.file_name
    graph["project_name"] = example.project_name

    return graph


def process_holdout(
    data_extractor: GitProjectExtractor, rng: Optional[random.Random], holdout: str, dest_path: str
) -> None:
    with Pool(USE_CPU) as pool:
        project_graphs = pool.map(extract_graphs, data_extractor.get_examples(holdout))
    graphs = [graph for graph in project_graphs if graph is not None]
    if rng:
        rng.shuffle(graphs)

    os.makedirs(dest_path, exist_ok=True)
    output_file = os.path.join(dest_path, f"graphs_{holdout}.jsonl.gz")
    print(f"Saving graph in JSONL format to {output_file}")
    with gzip.open(output_file, "wb") as out:
        for graph in tqdm(graphs):
            out.write((json.dumps(graph) + "\n").encode("utf-8"))


def preprocess(data_path: str, dest_path: str, random_seed: int, val_part: Optional[float], test_part: Optional[float]):
    data_extractor = GitProjectExtractor(data_path, random_seed, val_part, test_part)
    rng = random.Random(random_seed)

    if val_part:
        process_holdout(data_extractor, rng, "val", dest_path)
    if test_part:
        process_holdout(data_extractor, rng, "test", dest_path)
    process_holdout(data_extractor, rng, "train", dest_path)


if __name__ == "__main__":
    parser = configure_arg_parser()
    args = parser.parse_args()

    preprocess(args.data, args.dest, random_seed=args.seed, val_part=args.val_part, test_part=args.test_part)
