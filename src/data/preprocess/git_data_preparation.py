import collections
import glob
import json
import os
import random
import re
from typing import Tuple, Iterator, List, Dict, Optional

from tqdm.auto import tqdm

Example = collections.namedtuple("Example", ["language", "project_name", "file_name", "source_code"])

_DEFAULT_STATS_BOUNDARIES = {
    "Python": {"max_line_len": (37, 741), "content_len": (111, 42476)},
    "Java": {"max_line_len": (56, 177), "content_len": (305, 48661)},
    "Kotlin": {"max_line_len": (25, 158), "content_len": (69, 20402)},
}
_BAD_TEXT_REGEX = re.compile(r"auto[- ]?generated file", flags=re.IGNORECASE)


class GitProjectExtractor:
    def __init__(self, raw_data_path: str, random_seed: int, val_part: Optional[float], test_part: Optional[float]):
        self._path: str = raw_data_path
        self._rng: random.Random = random.Random(random_seed)

        self._val_part: float = val_part if val_part is not None else 0.0
        self._test_part: float = test_part if test_part is not None else 0.0
        assert self._val_part + self._test_part <= 1.0
        self._train_part: float = 1.0 - self._val_part - self._test_part

        self._processed_projects: Optional[Dict[str, List[List[Tuple[str, str, str, str]]]]] = None

    # Main method
    def get_examples(self, holdout: str, languages: Tuple[str] = ("Python",)) -> Iterator[Example]:
        """Read all files in specified language from dataset and return a project iterator"

        :param raw_data_path: path to the "dataset/v3" directory
        :param holdout: which holdout to return. Can be either "train", "val" and "test"
        :param languages: programming languages, projects in which will be searched
        :return: Iterator, which returns projects - Lists of Tuples, each of which represent project's files
        """
        if self._processed_projects is None:
            print(f"Extracting projects metainfo...")
            self._extract_projects(languages)
        return self._generate_examples_iter(holdout)

    # -------------------------------------- Stage methods -------------------------------------- #
    def _extract_projects(self, languages: Tuple[str]):
        lang_files = self._get_lang_files(languages)
        projects = self._get_files_projects(lang_files)
        found_projects_amount = len(projects)
        processed_projects, skipped_projects, found_files_amount = self._process_projects(projects)

        self._processed_projects = dict()
        self._rng.shuffle(processed_projects)
        train_projects_amount = int(self._train_part * len(processed_projects))
        val_projects_amount = int(self._val_part * len(processed_projects))
        self._processed_projects["train"] = processed_projects[:train_projects_amount]
        self._processed_projects["val"] = processed_projects[
            train_projects_amount : train_projects_amount + val_projects_amount
        ]
        self._processed_projects["test"] = processed_projects[train_projects_amount + val_projects_amount :]

        tqdm.write(
            f"Found {found_projects_amount} projects with {found_files_amount} files, "
            f"skipped {len(skipped_projects)} projects\n"
        )
        if len(skipped_projects) != 0:
            tqdm.write(f"Skipped projects: {skipped_projects}\n")

    def _generate_examples_iter(self, holdout: str) -> Iterator[Example]:
        """Yield all project files, one project at a time"""

        def read_file(path):
            with open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()

        assert self._processed_projects is not None
        for project in tqdm(self._processed_projects[holdout], desc=f"Reading {holdout} projects..."):
            examples = (
                Example(language, proj_name, filename, read_file(path))
                for language, proj_name, filename, path in project
            )
            yield from (
                example
                for example in examples
                if GitProjectExtractor._is_good_example(example.language, example.file_name, example.source_code)
            )

    @staticmethod
    def _is_good_example(language: str, filename: str, source_code: str) -> bool:
        if not filename or not source_code:
            return False

        # Check stats
        if not (
            _DEFAULT_STATS_BOUNDARIES[language]["content_len"][0]
            <= len(source_code)
            <= _DEFAULT_STATS_BOUNDARIES[language]["content_len"][1]
            and _DEFAULT_STATS_BOUNDARIES[language]["max_line_len"][0]
            <= max(len(line) for line in source_code.split("\n"))
            <= _DEFAULT_STATS_BOUNDARIES[language]["max_line_len"][1]
        ):
            return False

        # Regex check
        if re.search(_BAD_TEXT_REGEX, source_code):
            return False

        return True

    # --------------------------------- Paths processing methods -------------------------------- #
    def _get_lang_files(self, languages: Tuple[str]) -> List[Tuple[str, str]]:
        res: List[Tuple[str, str]] = []
        for language in languages:
            lang_files = glob.glob(
                os.path.join(self._path, "languages", language, ".*", "*", "*", "**", "*.*"), recursive=True
            )
            assert lang_files, f"There are no files in {self._path} with language {language}"
            print(f"Found {len(lang_files)} files' metainfos for {language} lang")
            res.extend((lang_file, language) for lang_file in lang_files)
        return res

    @staticmethod
    def _get_files_projects(lang_files: List[Tuple[str, str]]) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Group all files by projects"""
        projects = collections.defaultdict(list)
        for (file, lang) in lang_files:
            if os.path.isfile(file):
                project_name = os.sep.join(file.split(os.sep)[-3:-1])
                projects[project_name].append((file, lang))

        return list(projects.items())

    def _process_projects(
        self, projects: List[Tuple[str, List[Tuple[str, str]]]]
    ) -> Tuple[List[List[Tuple[str, str, str, str]]], List[str], int]:
        """Search for projects, extract real project names from dataset

        :param raw_data_path: path to the "dataset/v3" directory
        :param projects: output of _get_files_projects.

        :return: a Tuple,
            first item of which is a List, each item of which represents a single GitHub project
                and is itself a List, each item of which represents a single file in the project
                which is written in the specified language
                and is itself a Tuple, first item of which is the path to a file in the project structure,
                the second one is the path to the file in our dataset structure
                the third one is the language of the file.
            second item is the length of projects list.
        """
        processed_projects = []
        skipped_projects = []
        files_amount = 0
        for project_name, files in projects:
            author, repo, branch, filename = files[0][0].split(os.sep)[-4:]
            paths_dict_path = os.path.join(
                self._path,
                "repositories",
                author,
                repo,
                branch,
                "paths.json",
            )
            if os.path.exists(paths_dict_path):
                with open(paths_dict_path, "rt") as f:
                    paths_dict = json.load(f)

                names_and_paths = []
                for (file, lang) in files:
                    if os.path.basename(file) in paths_dict:
                        names_and_paths.append((lang, project_name, paths_dict[os.path.basename(file)], file))

                processed_projects.append(names_and_paths)
                files_amount += len(names_and_paths)
            else:
                skipped_projects.append(f"{author}/{repo}")

        return processed_projects, skipped_projects, files_amount