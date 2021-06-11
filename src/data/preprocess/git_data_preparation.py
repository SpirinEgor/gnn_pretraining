import collections
import glob
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Tuple, Iterator, List, Dict, Optional

from tqdm.auto import tqdm


@dataclass
class Example:
    language: str
    project_name: str
    file_name: str
    source_code: str


_DEFAULT_STATS_BOUNDARIES = {
    "Python": {"max_line_len": (37, 741), "content_len": (111, 42476)},
    "Java": {"max_line_len": (56, 177), "content_len": (305, 48661)},
    "Kotlin": {"max_line_len": (25, 158), "content_len": (69, 20402)},
}
_BAD_TEXT_REGEX = re.compile(r"auto[- ]?generated file", flags=re.IGNORECASE)
_BUCKET_SIZE = 1_000_000


class GitProjectExtractor:
    def __init__(
        self,
        raw_data_path: str,
        random_seed: int,
        val_part: Optional[float],
        test_part: Optional[float],
        languages: Tuple[str] = ("Python",),
    ):
        self._path: str = raw_data_path
        self._rng: random.Random = random.Random(random_seed)
        self._found_files_amount: Optional[int] = None

        self._holdout_sizes: Dict[str, float] = dict()
        self._holdout_sizes["val"] = val_part if val_part is not None else 0.0
        self._holdout_sizes["test"] = test_part if test_part is not None else 0.0
        assert self._holdout_sizes["val"] + self._holdout_sizes["test"] <= 1.0
        self._holdout_sizes["train"] = 1.0 - self._holdout_sizes["val"] - self._holdout_sizes["test"]

        self._processed_projects: Optional[Dict[str, List[List[Tuple[str, str, str, str]]]]] = None

        print(f"Extracting projects metainfo...")
        self._extract_projects(languages)

    def get_num_examples(self, holdout: str) -> int:
        assert self._found_files_amount is not None
        return int(self._found_files_amount * self._holdout_sizes[holdout])

    # Main method
    def get_examples(self, holdout: str) -> Iterator[Example]:
        """Read all files in specified language from dataset and return a project iterator"

        :param holdout: which holdout to return. Can be either "train", "val" and "test"
        :return: Iterator, which returns projects - Lists of Tuples, each of which represent project's files
        """
        return self._generate_examples_iter(holdout)

    # -------------------------------------- Stage methods -------------------------------------- #
    def _extract_projects(self, languages: Tuple[str]):
        lang_files = self._get_lang_files(languages)
        projects = self._get_files_projects(lang_files)
        found_projects_amount = len(projects)
        (
            processed_projects,
            skipped_projects,
            self._found_files_amount,
        ) = self._process_projects(projects)

        self._processed_projects = dict()
        self._rng.shuffle(processed_projects)
        train_projects_amount = int(self._holdout_sizes["train"] * len(processed_projects))
        val_projects_amount = int(self._holdout_sizes["val"] * len(processed_projects))
        self._processed_projects["train"] = processed_projects[:train_projects_amount]
        self._processed_projects["val"] = processed_projects[
            train_projects_amount : train_projects_amount + val_projects_amount
        ]
        self._processed_projects["test"] = processed_projects[train_projects_amount + val_projects_amount :]

        print(
            f"Found {found_projects_amount} projects with {self._found_files_amount} files, "
            f"skipped {len(skipped_projects)} projects\n"
        )
        if len(skipped_projects) != 0:
            print(f"Skipped projects: {skipped_projects}\n")

    def _generate_examples_iter(self, holdout: str) -> Iterator[Example]:
        """Yield all project files, one project at a time"""

        def read_file(path):
            with open(path, "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()

        bucket_to_shuffle: List[Example] = []

        assert self._processed_projects is not None
        for project in self._processed_projects[holdout]:
            examples = (
                Example(language, proj_name, filename, read_file(path))
                for language, proj_name, filename, path in project
            )
            bucket_to_shuffle.extend(
                example
                for example in examples
                if GitProjectExtractor._is_good_example(example.language, example.file_name, example.source_code)
            )
            if len(bucket_to_shuffle) > _BUCKET_SIZE:
                self._rng.shuffle(bucket_to_shuffle)
                yield from bucket_to_shuffle
                bucket_to_shuffle = []

        yield from bucket_to_shuffle

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
                os.path.join(
                    self._path,
                    "languages",
                    language,
                    ".*",
                    "*",
                    "*",
                    "**",
                    "*.*",
                ),
                recursive=True,
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
                        names_and_paths.append(
                            (
                                lang,
                                project_name,
                                paths_dict[os.path.basename(file)],
                                file,
                            )
                        )

                processed_projects.append(names_and_paths)
                files_amount += len(names_and_paths)
            else:
                skipped_projects.append(f"{author}/{repo}")

        return processed_projects, skipped_projects, files_amount
