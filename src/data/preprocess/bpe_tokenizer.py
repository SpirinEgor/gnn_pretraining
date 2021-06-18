import pickle
from argparse import ArgumentParser
from itertools import chain, repeat, islice
from typing import Counter, Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, StripAccents, NFKC, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.utils import UNK, MASK, PAD

DROPOUT = None
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = [PAD, UNK, MASK]
BATCH_SIZE = 40960


def batch_iterator(subtokens_counter: Counter[str]) -> Iterator[Iterator[str]]:
    chained_subtokens = chain(
        *(repeat(st.encode("utf-8", "ignore").decode("utf-8"), cnt) for st, cnt in subtokens_counter.items())
    )
    chain_iter = iter(chained_subtokens)
    return iter(lambda: tuple(islice(chain_iter, BATCH_SIZE)), ())


def train_bpe(vocabulary_path: str, output_path: str):
    bpe_tokenizer = Tokenizer(BPE(dropout=DROPOUT, unk_token=UNK, fuse_unk=True))
    bpe_tokenizer.normalizer = Sequence([NFKC(), Lowercase(), StripAccents()])
    bpe_tokenizer.pre_tokenizer = ByteLevel()

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)

    with open(vocabulary_path, "rb") as vf:
        token_counter = pickle.load(vf)

    length = sum([cnt for _, cnt in token_counter.items()])
    bpe_tokenizer.train_from_iterator(batch_iterator(token_counter), trainer=trainer, length=length)

    bpe_tokenizer.save(output_path)


if __name__ == "__main__":
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-v", "--vocab", help="Path to file with collected subtokens", required=True)
    __arg_parser.add_argument("-o", "--output", help="Path to output file", required=True)
    __args = __arg_parser.parse_args()

    train_bpe(__args.vocab, __args.output)
