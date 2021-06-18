import pickle
from collections import Counter
from typing import Optional, Tuple

from src.utils import UNK, MASK, PAD


class Vocabulary:
    def __init__(self, vocabulary_path: str, size: Optional[int] = None):
        with open(vocabulary_path, "rb") as vocab_file:
            token_counter: Counter[str] = pickle.load(vocab_file)
        self.__pad_id, self.__unk_id, self.__mask_id = 0, 1, 2
        self.__token2id = {PAD: self.__pad_id, UNK: self.__unk_id, MASK: self.__mask_id}
        self.__token2id.update(((t, i + 3) for i, (t, _) in enumerate(token_counter.most_common(size))))

    @property
    def pad(self) -> Tuple[str, int]:
        return PAD, self.__pad_id

    @property
    def unk(self) -> Tuple[str, int]:
        return UNK, self.__unk_id

    @property
    def mask(self) -> Tuple[str, int]:
        return MASK, self.__mask_id

    def __getitem__(self, item: str):
        return self.__token2id.get(item, self.__unk_id)

    def __len__(self) -> int:
        return len(self.__token2id)
