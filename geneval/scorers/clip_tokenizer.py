"""CLIP byte-level BPE tokenizer (same algorithm and vocab file as the original CLIP release),
kept in-repo so the ONNX scorer has no torch / HF dependency."""
from __future__ import annotations

import gzip
import html
import pathlib
from functools import lru_cache

import ftfy
import numpy as np
import regex as re

VOCAB = pathlib.Path(__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(word, word[1:])}


def _clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip().lower()


class ClipTokenizer:
    def __init__(self, path: pathlib.Path = VOCAB):
        self.byte_encoder = bytes_to_unicode()
        merges = gzip.open(path).read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        vocab += ["".join(m) for m in merges] + ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)
        self.sot = self.encoder["<|startoftext|>"]
        self.eot = self.encoder["<|endoftext|>"]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if bigram not in self.ranks:
                break
            a, b = bigram
            out, i = [], 0
            while i < len(word):
                try:
                    j = word.index(a, i)
                except ValueError:
                    out.extend(word[i:])
                    break
                out.extend(word[i:j])
                i = j
                if word[i] == a and i < len(word) - 1 and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        res = " ".join(word)
        self.cache[token] = res
        return res

    def encode(self, text: str) -> list[int]:
        ids = []
        for tok in re.findall(self.pat, _clean(text)):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(tok).split(" "))
        return ids

    def __call__(self, texts: list[str], context: int = 77) -> np.ndarray:
        out = np.zeros((len(texts), context), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.sot] + self.encode(t)[: context - 2] + [self.eot]
            out[i, : len(ids)] = ids
        return out
