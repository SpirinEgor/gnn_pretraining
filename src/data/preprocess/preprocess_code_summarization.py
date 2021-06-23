import ast
from argparse import ArgumentParser
from typing import Iterable

from datasets import load_dataset, tqdm

from src.data.preprocess.example import Example
from src.data.preprocess.graph_extraction import process_data, logger


def __vocabulary_func(graph: dict) -> Iterable[str]:
    return (graph["docstring"],)


# Took from https://gist.github.com/phpdude/1ae6f19de213d66286c8183e9e3b9ec1
def __remove_docs(source_code: str) -> str:
    parsed = ast.parse(source_code)
    for node in ast.walk(parsed):
        # let's work only on functions & classes definitions
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue
        if not len(node.body):
            continue
        if not isinstance(node.body[0], ast.Expr):
            continue
        if not hasattr(node.body[0], "value") or not isinstance(node.body[0].value, ast.Str):
            continue

        # Uncomment lines below if you want print what and where we are removing
        # print(node)
        # print(node.body[0].value.s)
        node.body = node.body[1:]
        # add "pass" statement here
        if len(node.body) < 1:
            node.body.append(ast.Pass())

    return ast.unparse(parsed)


class RemoveDocsComments(ast.NodeTransformer):
    def visit_Assign(self, node):
        return None

    def visit_AugAssign(self, node):
        return None


def preprocess(save_path_dir: str, lang: str = "python") -> None:
    assert lang in (
        "go",
        "java",
        "javascript",
        "php",
        "python",
        "ruby",
    ), f"Language {lang} is not presented in CodeXGLUE"
    assert lang == "python", f"Language {lang} is not supported yet"

    dataset = load_dataset("code_x_glue_ct_code_to_text", lang)

    for holdout in ("train", "validation", "test"):
        dataset_split = dataset[holdout]
        examples = []
        for datapoint in tqdm(dataset_split, desc="Reading examples"):
            try:
                source_code = __remove_docs(datapoint["code"])

                examples.append(
                    Example(
                        language=lang,
                        project_name=datapoint["repo"],
                        file_name=datapoint["path"],
                        source_code=source_code,
                        docstring=" ".join(datapoint["docstring_tokens"]),
                    )
                )
            except SyntaxError as e:
                logger.warning(f"Failed to remove doc: {e}")

        process_data(examples, holdout, save_path_dir, __vocabulary_func if holdout == "train" else None)


if __name__ == "__main__":

    def main():
        arg_parser = ArgumentParser()
        arg_parser.add_argument(
            "-o",
            "--save_path",
            type=str,
            required=True,
            help="Path to directory where graphs will be stored",
        )
        arg_parser.add_argument(
            "-l",
            "--lang",
            type=str,
            required=False,
            default="python",
            help="Path to directory where graphs will be stored",
        )
        args = arg_parser.parse_args()

        preprocess(args.save_path, args.lang)

    main()
