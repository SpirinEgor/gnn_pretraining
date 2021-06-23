import json
from argparse import ArgumentParser
from itertools import chain, repeat
from typing import Counter

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, StripAccents, NFKC, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.utils import UNK, MASK, PAD

DROPOUT = None
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = [PAD, UNK, MASK]


def batch_iterator(subtokens_counter: Counter[str]) -> chain[str]:
    return chain(*(repeat(st.encode("utf-8", "ignore").decode("utf-8"), cnt) for st, cnt in subtokens_counter.items()))


def train_bpe(vocabulary_path: str, output_path: str, reversible: bool):
    bpe_tokenizer = Tokenizer(BPE(dropout=DROPOUT, unk_token=UNK, fuse_unk=True))
    if not reversible:
        # Don't care if we can't generate the same text as in dataset
        bpe_tokenizer.normalizer = Sequence([NFKC(), Lowercase(), StripAccents()])
    else:
        bpe_tokenizer.normalizer = Sequence([NFKC()])
    bpe_tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    with open(vocabulary_path, "r") as vf:
        token_counter = json.load(vf)

    length = sum([cnt for _, cnt in token_counter.items()])
    bpe_tokenizer.train_from_iterator(batch_iterator(token_counter), trainer=trainer, length=length)

    bpe_tokenizer.save(output_path)


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-v", "--vocab", help="Path to file with collected subtokens", required=True)
    __arg_parser.add_argument("-o", "--output", help="Path to output file", required=True)
    __arg_parser.add_argument(
        "--reversible", help="Whether it's important to save ability to generate sequence", action="store_true"
    )
    __args = __arg_parser.parse_args()

    train_bpe(__args.vocab, __args.output, __args.reversible)
