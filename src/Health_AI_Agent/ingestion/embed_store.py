import logging
from typing import Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from app.config import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_USE_FLASH_ATTENTION,
)

logging.basicConfig(level=logging.INFO)


class ClinicalTransformerEmbedder:
    """
    Flash Attention-enabled Clinical Transformer embedder.

    The model produces pooled clinical-text representations. The pipeline
    emits 256-dimensional vectors by default so they can be stored directly in
    BigQuery as ARRAY<FLOAT64>.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        embedding_dim: int = EMBEDDING_DIMENSION,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            if EMBEDDING_USE_FLASH_ATTENTION:
                model_kwargs["attn_implementation"] = "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        try:
            self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        except (ImportError, TypeError, ValueError) as exc:
            logging.warning(
                "Model does not accept Flash Attention options; loading "
                "without attn_implementation. Reason: %s",
                exc,
            )
            self.model = AutoModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> List[List[float]]:
        """
        Encode text records into normalized 256-dimensional embeddings.
        """

        text_list = [text or "" for text in texts]
        embeddings: List[List[float]] = []

        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]

            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}

            with torch.inference_mode():
                outputs = self.model(**tokens)
                pooled = self._mean_pool(
                    outputs.last_hidden_state,
                    tokens["attention_mask"],
                )
                projected = self._to_dimension(pooled, self.embedding_dim)
                normalized = F.normalize(projected, p=2, dim=1)

            embeddings.extend(normalized.float().cpu().tolist())

        return embeddings

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        masked_embeddings = token_embeddings * expanded_mask
        summed = masked_embeddings.sum(dim=1)
        counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _to_dimension(embeddings: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Convert model hidden size to the requested embedding dimension.

        ClinicalBERT-family models commonly emit 768 dimensions. For this
        pipeline we keep a deterministic 256-dimensional representation by
        taking the leading dimensions after pooling. If a future clinical model
        has a native 256-d hidden size, this becomes a no-op.
        """

        hidden_size = embeddings.shape[1]

        if hidden_size == embedding_dim:
            return embeddings

        if hidden_size > embedding_dim:
            return embeddings[:, :embedding_dim]

        padding = torch.zeros(
            embeddings.shape[0],
            embedding_dim - hidden_size,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        return torch.cat([embeddings, padding], dim=1)
