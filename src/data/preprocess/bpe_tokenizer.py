import pickle
from argparse import ArgumentParser
from typing import List, Generator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, StripAccents, NFKC, Strip, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.utils import UNK, MASK, PAD

DROPOUT = None
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = [PAD, UNK, MASK]
BATCH_SIZE = 1024


def batch_iterator(subtokens: List[str]) -> Generator[List[str], None, None]:
    for i in range(0, len(subtokens), BATCH_SIZE):
        yield [st.encode("utf-8", "ignore").decode("utf-8") for st in subtokens[i : i + BATCH_SIZE]]


def train_bpe(vocabulary_path: str, output_path: str):
    bpe_tokenizer = Tokenizer(BPE(dropout=DROPOUT, unk_token=UNK, fuze_unk=True))
    bpe_tokenizer.normalizer = Sequence([NFKC(), Lowercase(), StripAccents()])
    bpe_tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    with open(vocabulary_path, "rb") as vf:
        token_counter = pickle.load(vf)

    subtokens = list(token_counter.keys())
    bpe_tokenizer.train_from_iterator(batch_iterator(subtokens), trainer=trainer, length=len(subtokens))

    bpe_tokenizer.save(output_path)


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-v", "--vocab", help="Path to file with collected subtokens", required=True)
    __arg_parser.add_argument("-o", "--output", help="Path to output file", required=True)
    __args = __arg_parser.parse_args()

    train_bpe(__args.vocab, __args.output)
