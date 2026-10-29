"""Pure-Python BERT (uncased) WordPiece tokenizer that reads a Hugging Face tokenizer.json.

Used for the ONNX sentence encoders (all-MiniLM-L6-v2, bge-small-en-v1.5) and the
ms-marco cross-encoder, so that no tokenizer library is needed. It follows the reference
BertTokenizer: clean text, split CJK characters, lower-case, strip accents, split on
punctuation, then greedy longest-match WordPiece with the "##" continuation prefix.
"""
import json
import unicodedata

import numpy as np


def _is_whitespace(ch):
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch):
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punctuation(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
            or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
            or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F))


class WordPiece:
    def __init__(self, tokenizer_json, lowercase=True, max_chars_per_word=100):
        with open(tokenizer_json, encoding="utf-8") as fh:
            tj = json.load(fh)
        self.vocab = tj["model"]["vocab"]
        self.unk = tj["model"].get("unk_token", "[UNK]")
        self.prefix = tj["model"].get("continuing_subword_prefix", "##")
        self.lowercase = lowercase
        self.max_chars = max_chars_per_word
        self.cls, self.sep, self.pad = self.vocab["[CLS]"], self.vocab["[SEP]"], self.vocab["[PAD]"]
        self.unk_id = self.vocab[self.unk]
        self._cache = {}

    def _basic(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_cjk(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        words = []
        for tok in "".join(out).split():
            if self.lowercase:
                tok = tok.lower()
                tok = "".join(c for c in unicodedata.normalize("NFD", tok) if unicodedata.category(c) != "Mn")
            cur = []
            for ch in tok:
                if _is_punctuation(ch):
                    if cur:
                        words.append("".join(cur))
                        cur = []
                    words.append(ch)
                else:
                    cur.append(ch)
            if cur:
                words.append("".join(cur))
        return words

    def _wordpiece(self, word):
        if word in self._cache:
            return self._cache[word]
        if len(word) > self.max_chars:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = self.prefix + sub
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                ids = [self.unk_id]
                break
            ids.append(cur)
            start = end
        if len(self._cache) < 200000:
            self._cache[word] = ids
        return ids

    def tokenize_ids(self, text):
        ids = []
        for w in self._basic(text or ""):
            ids.extend(self._wordpiece(w))
        return ids

    def encode(self, text, max_len=256):
        ids = self.tokenize_ids(text)[: max_len - 2]
        return [self.cls] + ids + [self.sep]

    def encode_pair(self, a, b, max_len=512):
        ia, ib = self.tokenize_ids(a), self.tokenize_ids(b)
        while len(ia) + len(ib) > max_len - 3:          # longest-first truncation
            if len(ia) >= len(ib):
                ia.pop()
            else:
                ib.pop()
        ids = [self.cls] + ia + [self.sep] + ib + [self.sep]
        tt = [0] * (len(ia) + 2) + [1] * (len(ib) + 1)
        return ids, tt

    def batch(self, seqs, types=None):
        n = max(len(s) for s in seqs)
        ids = np.full((len(seqs), n), self.pad, dtype=np.int64)
        mask = np.zeros((len(seqs), n), dtype=np.int64)
        tt = np.zeros((len(seqs), n), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
            if types is not None:
                tt[i, : len(s)] = types[i]
        return ids, mask, tt
