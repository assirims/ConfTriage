"""ONNX sentence encoders (bi-encoders) and the ms-marco cross-encoder, run with onnxruntime on CPU."""
import os

import numpy as np
import onnxruntime as ort

from .wordpiece import WordPiece


def _session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


class BiEncoder:
    """pooling: 'mean' (all-MiniLM-L6-v2) or 'cls' (bge-small-en-v1.5). Output vectors are L2-normalised."""

    def __init__(self, model_dir, pooling="mean", max_len=256, query_prefix="", threads=2):
        self.tok = WordPiece(os.path.join(model_dir, "tokenizer.json"))
        self.sess = _session(os.path.join(model_dir, "onnx", "model.onnx"), threads)
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.pooling, self.max_len, self.query_prefix = pooling, max_len, query_prefix

    def encode(self, texts, batch_size=32, is_query=False):
        texts = [(self.query_prefix + t) if is_query else t for t in texts]
        enc = [self.tok.encode(t, self.max_len) for t in texts]
        order = np.argsort([len(e) for e in enc])            # length bucketing for speed
        out = np.zeros((len(texts), 0), dtype=np.float32)
        vecs = [None] * len(texts)
        for s in range(0, len(order), batch_size):
            idx = order[s: s + batch_size]
            ids, mask, tt = self.tok.batch([enc[i] for i in idx])
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = tt
            h = self.sess.run(None, feed)[0]                   # last_hidden_state (B, T, H)
            if self.pooling == "cls":
                v = h[:, 0, :]
            else:
                m = mask[:, :, None].astype(np.float32)
                v = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
            for j, i in enumerate(idx):
                vecs[i] = v[j]
        out = np.vstack(vecs).astype(np.float32)
        return out


class CrossEncoder:
    def __init__(self, model_dir, max_len=512, threads=2):
        self.tok = WordPiece(os.path.join(model_dir, "tokenizer.json"))
        self.sess = _session(os.path.join(model_dir, "onnx", "model.onnx"), threads)
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.max_len = max_len

    def score(self, queries, passages, batch_size=16):
        pairs = [self.tok.encode_pair(q, p, self.max_len) for q, p in zip(queries, passages)]
        scores = np.zeros(len(pairs), dtype=np.float32)
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s: s + batch_size]
            ids, mask, tt = self.tok.batch([c[0] for c in chunk], [c[1] for c in chunk])
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = tt
            scores[s: s + len(chunk)] = self.sess.run(None, feed)[0].reshape(-1)
        return scores
