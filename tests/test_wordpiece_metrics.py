"""Unit tests: WordPiece tokenizer against known BERT-uncased tokenisations, and ranking metrics."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MINILM  # noqa: E402
from lib.metrics import holm, mcnemar_exact, rank_of, reciprocal_rank_fusion, summarise_ranks  # noqa: E402
from lib.wordpiece import WordPiece  # noqa: E402


def test_wordpiece_known_examples():
    tok = WordPiece(os.path.join(MINILM, "tokenizer.json"))
    inv = {v: k for k, v in tok.vocab.items()}
    toks = [inv[i] for i in tok.tokenize_ids("UNwantéd,running")]
    assert toks == ["unwanted", ",", "running"], toks      # example from the BERT reference tests
    toks = [inv[i] for i in tok.tokenize_ids("unaffable")]
    assert toks == ["una", "##ffa", "##ble"], toks          # bert-base-uncased vocabulary
    ids = tok.encode("Hello world")
    assert ids[0] == tok.cls and ids[-1] == tok.sep
    pair, tt = tok.encode_pair("a b c", "d e", max_len=6)
    assert len(pair) == 6 and tt.count(1) >= 2


def test_rank_metrics():
    assert rank_of("b", ["a", "b", "c"]) == 2
    assert np.isinf(rank_of("z", ["a"]))
    s = summarise_ranks([1, 2, np.inf, 11])
    assert abs(s["R@1"] - 0.25) < 1e-9 and abs(s["R@10"] - 0.5) < 1e-9
    assert abs(s["MRR"] - (1 + 0.5 + 0 + 1 / 11) / 4) < 1e-9


def test_fusion_and_tests():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "a"], ["b"]])[0] == "b"
    n10, n01, p = mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
    assert (n10, n01) == (3, 0) and abs(p - 0.25) < 1e-9
    assert list(holm([0.01, 0.04, 0.03])) == [0.03, 0.06, 0.06]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
