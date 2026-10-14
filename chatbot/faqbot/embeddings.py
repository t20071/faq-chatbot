"""Query encoder for vector search: all-MiniLM-L6-v2 run with ONNX Runtime.

Document vectors are built offline with sentence-transformers
(scripts/build_embeddings.py). At query time the app needs the same model, but
sentence-transformers depends on PyTorch, which is far too large for a Vercel
function. This class reproduces sentence-transformers' pipeline for this model
(tokenize -> transformer -> mean pooling -> L2 normalise) with onnxruntime and
tokenizers instead. build_embeddings.py checks that the two produce the same vectors.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from faqbot.config import EMBEDDING_MAX_TOKENS, MODEL_DIR, ONNX_MODEL_FILE


class MiniLMEncoder:
    def __init__(self, model_dir: Path = MODEL_DIR, max_tokens: int = EMBEDDING_MAX_TOKENS):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # serverless CPUs are small; avoid thread contention
        self.session = ort.InferenceSession(
            str(model_dir / ONNX_MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised float32 vector (384 dims) per text.

        Texts are encoded one at a time: the 8-bit model quantises activations with a
        scale computed over the whole batch (padding included), so batching would make
        a text's vector depend on its neighbours. Documents are embedded the same way.
        """
        return np.vstack([self._encode_batch([text]) for text in texts])

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        inputs = {k: v for k, v in inputs.items() if k in self.input_names}
        token_vectors = self.session.run(None, inputs)[0]  # (batch, tokens, 384)

        # Mean pooling over real tokens (padding excluded), as sentence-transformers does.
        mask = inputs["attention_mask"][..., None].astype(np.float32)
        pooled = (token_vectors * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)
