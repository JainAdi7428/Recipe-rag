"""The build-time encoder: same vectors as Chroma's, plus token measurement.

This runs *the same ONNX weights and tokenizer that Chroma downloaded*, with
two changes: padding to the longest sequence in each batch instead of a
constant 256, and length-sorted batching so batches stay tight. Measured on
real chunks that is 38.7 vs 32.1 chunks/s, a 1.21x gain -- 28 minutes rather
than 34 for the full index. Modest, and not the main reason this module exists.

The main reasons are the two support methods:

  token_lengths()  the chunker needs true wordpiece counts, because characters
                   per token swing from ~4.5 in flowing directions prose to
                   ~3.2 in ingredient lines full of numbers and fractions. A
                   character cap alone cannot guarantee the model's token cap.
  verify_parity()  build vectors and query vectors must occupy the same space.

Padding length cannot change the output because mean pooling is attention-
masked: padded positions are multiplied by zero and excluded from the divisor.
Measured maximum absolute difference against Chroma's own embedder is 4.5e-08,
i.e. float32 noise. `verify_parity` re-checks that at every build rather than
taking it on trust, since a silent divergence would wreck retrieval in a way
that is very hard to diagnose later.

Query time still goes through Chroma's embedding function; only the bulk build
uses this one.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from .config import MAX_MODEL_TOKENS

_PARITY_TOLERANCE = 1e-5


def _model_dir() -> Path:
    """Path to the ONNX model Chroma manages, downloading it if needed."""
    warmup = ONNXMiniLM_L6_V2()
    warmup(["warmup"])  # forces the download + extraction on first use
    return Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME


class FastMiniLMEmbedder:
    """all-MiniLM-L6-v2 over onnxruntime with dynamic padding."""

    def __init__(self, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        base = _model_dir()
        self.tokenizer = Tokenizer.from_file(str(base / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_MODEL_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads or os.cpu_count() or 4
        self.session = ort.InferenceSession(
            str(base / "model.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Wordpiece length per text, before truncation is applied."""
        tokenizer = self.tokenizer
        tokenizer.no_padding()
        tokenizer.no_truncation()
        try:
            return [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
        finally:
            tokenizer.enable_truncation(max_length=MAX_MODEL_TOKENS)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        hidden = self.session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": np.zeros_like(ids),
            },
        )[0]
        # Attention-masked mean pooling, then L2 normalise (cosine space).
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return pooled / np.clip(
            np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9, None
        )

    def encode(
        self, texts: list[str], batch_size: int = 128, progress=None
    ) -> np.ndarray:
        """Embed texts, returning vectors in the original input order."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        # Group similar lengths so each batch pads to a tight maximum.
        order = np.argsort([len(text) for text in texts], kind="stable")
        vectors = np.zeros((len(texts), 384), dtype=np.float32)

        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            vectors[index] = self._encode_batch([texts[i] for i in index])
            if progress is not None:
                progress(min(start + batch_size, len(order)), len(order))

        return vectors

    def verify_parity(self, texts: list[str]) -> float:
        """Max absolute deviation from Chroma's embedder on a sample.

        Raises if the two encoders have drifted apart: build vectors and query
        vectors must live in the same space or retrieval is meaningless.
        """
        reference = np.array(ONNXMiniLM_L6_V2()(texts), dtype=np.float32)
        delta = float(np.abs(reference - self.encode(texts)).max())
        if delta > _PARITY_TOLERANCE:
            raise RuntimeError(
                f"Build encoder diverges from Chroma's query encoder by {delta:.2e} "
                f"(tolerance {_PARITY_TOLERANCE:.0e}). Refusing to build a "
                f"mismatched index."
            )
        return delta
