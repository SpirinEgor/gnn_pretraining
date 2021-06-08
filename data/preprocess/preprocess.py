import json
import logging
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count, Lock, Semaphore, Queue
from os import getcwd
from os.path import join, dirname, basename
from typing import Dict, Optional

from tqdm.auto import tqdm

from data.preprocess.typilus.graphgenerator import AstGraphGenerator
from data.preprocess.typilus.type_lattice_generator import TypeLatticeGenerator

DELIMITER_EXAMPLE = "␢"
DELIMITER_FILENAME = "₣"
TYPE_LATTICE_CONFIG = join(getcwd(), dirname(__file__), "typilus/type_lattice_config.json")

USE_CPU = cpu_count()

LOG_FILENAME = "log.txt"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, filemode="w")
logger = logging.getLogger(__name__)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-d", "--data", help="Path to file with source code")
    return arg_parser


def extract_graph(file_name: str, source_code: str) -> Optional[Dict]:
    type_lattice = TypeLatticeGenerator(TYPE_LATTICE_CONFIG)
    try:
        visitor = AstGraphGenerator(source_code, type_lattice)
        graph = visitor.build()
    except Exception as e:
        # TODO: proper logging in multiprocessing
        logging.error(f"Can't generate graph from {file_name}, exception: {e}")
        return None
    if graph is None or len(graph["supernodes"]) == 0:
        logging.error(f"Found empty graph for {file_name}")
        return None
    graph["filename"] = file_name
    return graph


def preprocess(data_path: str):
    with open(data_path, encoding="utf-8", errors="ignore") as f:
        examples = [
            tuple(map(lambda x: x.strip(), e.split(DELIMITER_FILENAME))) for e in f.read().split(DELIMITER_EXAMPLE) if e
        ]

    with Pool(USE_CPU) as pool:
        graphs = pool.starmap(extract_graph, tqdm(examples))
    graphs = [g for g in graphs if g is not None]
    print(f"Extract {len(graphs)} from {len(examples)} source files")

    data_folder = dirname(data_path)
    data_name = basename(data_path).split(".")[0]
    output_file = join(data_folder, data_name + ".jsonl")
    print(f"Saving graph in JSONL format to {output_file}")
    with open(output_file, "w") as out:
        for graph in tqdm(graphs):
            out.write(json.dumps(graph) + "\n")


if __name__ == "__main__":
    parser = configure_arg_parser()
    args = parser.parse_args()

    preprocess(args.data)
