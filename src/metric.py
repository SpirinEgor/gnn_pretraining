import re
from math import log, exp
from sys import float_info
from typing import List, Dict, Tuple, Union, Any

import xml.sax.saxutils

import torch
from torchmetrics import Metric


class CodeXGlueBleu(Metric):
    __normalize_patterns_1 = [
        ("<skipped>", ""),  # strip "skipped" tags
        (r"-\n", ""),  # strip end-of-line hyphenation and join lines
        (r"\n", " "),  # join lines
        #    (r'(\d)\s+(?=\d)', r'\1'), # join digits
    ]
    __normalize1 = [(re.compile(pattern), replace) for (pattern, replace) in __normalize_patterns_1]

    __normalize_patterns_2 = [
        (r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 "),  # tokenize punctuation. apostrophe is missing
        (r"([^0-9])([\.,])", r"\1 \2 "),  # tokenize period and comma unless preceded by a digit
        (r"([\.,])([^0-9])", r" \1 \2"),  # tokenize period and comma unless followed by a digit
        (r"([0-9])(-)", r"\1 \2 "),  # tokenize dash when preceded by a digit
    ]
    __normalize2 = [(re.compile(pattern), replace) for (pattern, replace) in __normalize_patterns_2]

    def __init__(
        self,
        n_grams: int,
        smooth: int,
        normalize: bool = True,
        preserve_case: bool = False,
        eff_ref_len: str = "shortest",
    ):
        super().__init__()
        self.__n_grams = n_grams
        self.__smooth = smooth
        self.__is_normalized = normalize
        self.__preserve_case = preserve_case
        self.__eff_ref_len = eff_ref_len

        self.add_state("score", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")

    def __normalize(self, text: str) -> List[str]:
        # Added to bypass NIST-style pre-processing of hyp and ref files -- wade
        if not self.__is_normalized:
            return text.split()
        if type(text) is not str:
            text = " ".join(text)
        # Language-independent part:
        for (pattern, replace) in self.__normalize1:
            text = re.sub(pattern, replace, text)
        text = xml.sax.saxutils.unescape(text, {"&quot;": '"'})
        # Language-dependent part (assuming Western languages):
        text = " %s " % text
        if not self.__preserve_case:
            text = text.lower()  # this might not be identical to the original
        for (pattern, replace) in self.__normalize2:
            text = re.sub(pattern, replace, text)
        return text.split()

    def __count_ngrams(self, words: List[str]) -> Dict[Tuple[str, ...], int]:
        counts: Dict[Tuple[str, ...], int] = {}
        for k in range(1, self.__n_grams + 1):
            for i in range(len(words) - k + 1):
                n_gram = tuple(words[i : i + k])
                counts[n_gram] = counts.get(n_gram, 0) + 1
        return counts

    def __cook_references(self, references: List[str]) -> Tuple[List[int], Dict[Tuple[str, ...], int]]:
        """Takes a list of reference sentences for a single segment
        and returns an object that encapsulates everything that BLEU
        needs to know about them.
        """
        max_counts: Dict[Tuple[str, ...], int] = {}
        for ref in references:
            counts = self.__count_ngrams(self.__normalize(ref))
            for (n_gram, count) in counts.items():
                max_counts[n_gram] = max(max_counts.get(n_gram, 0), count)
        return [len(ref) for ref in references], max_counts

    def __cook_hypothesis(self, hypothesis: str, references: Tuple[List[int], Dict[Tuple[str, ...], int]]):
        """Takes a test sentence and returns an object that
        encapsulates everything that BLEU needs to know about it."""
        (ref_lens, ref_max_counts) = references
        hypothesis_words = self.__normalize(hypothesis)
        result: Dict[str, Any] = {"test_len": len(hypothesis_words)}

        # Calculate effective reference sentence length.
        if self.__eff_ref_len == "shortest":
            result["ref_len"] = min(ref_lens)
        elif self._eff_ref_len == "average":
            result["ref_len"] = float(sum(ref_lens)) / len(ref_lens)
        elif self._eff_ref_len == "closest":
            min_diff = None
            for ref_len in ref_lens:
                if min_diff is None or abs(ref_len - len(hypothesis_words)) < min_diff:
                    min_diff = abs(ref_len - len(hypothesis_words))
                    result["ref_len"] = ref_len
        else:
            raise NotImplementedError(f"Unknown value for effective reference sentence length: {self.__eff_ref_len}")

        result["guess"] = [max(len(hypothesis_words) - k + 1, 0) for k in range(1, self.__n_grams + 1)]

        result["correct"] = [0] * self.__n_grams
        counts = self.__count_ngrams(hypothesis_words)
        for (n_gram, count) in counts.items():
            result["correct"][len(n_gram) - 1] += min(ref_max_counts.get(n_gram, 0), count)

        return result

    def __score_cooked(self, all_comps):
        total_comps = {"test_len": 0, "ref_len": 0, "guess": [0] * self.__n_grams, "correct": [0] * self.__n_grams}
        for comps in all_comps:
            for key in ["test_len", "ref_len"]:
                total_comps[key] += comps[key]
            for key in ["guess", "correct"]:
                for k in range(self.__n_grams):
                    total_comps[key][k] += comps[key][k]

        log_bleu = 0.0
        all_bleus = []
        for k in range(self.__n_grams):
            correct = total_comps["correct"][k]
            guess = total_comps["guess"][k]
            add_smooth = 0
            if self.__smooth == 1 and k > 0:
                add_smooth = 1
            log_bleu += log(correct + add_smooth + float_info.min) - log(guess + add_smooth + float_info.min)
            if guess == 0:
                all_bleus.append(-10000000)
            else:
                all_bleus.append(log(correct + float_info.min) - log(guess))

        log_bleu /= float(self.__n_grams)
        all_bleus.insert(0, log_bleu)

        brev_penalty = min(0.0, 1 - float(total_comps["ref_len"] + 1) / (total_comps["test_len"] + 1))
        for i in range(len(all_bleus)):
            if i == 0:
                all_bleus[i] += brev_penalty
            all_bleus[i] = exp(all_bleus[i])
        return all_bleus

    def __bleu(self, references: List[str], hypothesis: str) -> List[float]:
        cooked_refs = self.__cook_references(references)
        test = self.__cook_hypothesis(hypothesis, cooked_refs)
        return self.__score_cooked([test])

    def update(self, hypotheses: List[str], references: List[List[str]]):  # type: ignore
        for hypothesis, reference in zip(hypotheses, references):
            bleu = self.__bleu(reference, hypothesis)
            self.score += bleu[0]  # type: ignore
            self.num += 1  # type: ignore

    def compute(self):
        return self.score * 100 / self.num
