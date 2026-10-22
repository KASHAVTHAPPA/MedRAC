"""Sentence embeddings with sentence-transformers/all-MiniLM-L6-v2.

The model runs through ONNX Runtime instead of PyTorch. The quantised model is
23 MB and needs no GPU, which keeps the app small enough for a serverless
function. The steps are exactly what sentence-transformers does internally:

    tokens -> transformer -> mean pooling over real tokens -> L2 normalisation

Because every vector has length 1, cosine similarity is just a dot product.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

MAX_TOKENS = 256  # the model was trained with this window


class TextEncoder(Protocol):
    """Anything that turns texts into unit-length vectors (tests use a fake)."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class MiniLMEmbedder:
    dim = 384

    def __init__(self, model_dir: Path, threads: int = 0) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / "model_quint8_avx2.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = [self._encode_batch(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(out)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)

        hidden = self.session.run(None, feeds)[0]  # (batch, tokens, 384)

        # Mean pooling: average the token vectors, ignoring padding.
        m = mask[:, :, None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return _normalize(pooled)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype(np.float32)
