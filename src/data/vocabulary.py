import pickle
from collections import Counter
from typing import Optional, Tuple


class Vocabulary:

    __PAD = "<PAD>"
    __UNK = "<UNK>"

    def __init__(self, vocabulary_path: str, size: Optional[int] = None):
        with open(vocabulary_path, "rb") as vocab_file:
            token_counter: Counter[str] = pickle.load(vocab_file)
        self.__pad_id, self.__unk_id = 0, 1
        self.__token2id = {self.__PAD: self.__pad_id, self.__UNK: self.__unk_id}
        self.__token2id.update(((t, i + 2) for i, (t, _) in enumerate(token_counter.most_common(size))))

    @property
    def pad(self) -> Tuple[str, int]:
        return self.__PAD, self.__pad_id

    @property
    def unk(self) -> Tuple[str, int]:
        return self.__UNK, self.__unk_id

    def __getitem__(self, item: str):
        return self.__token2id.get(item, self.__unk_id)
