import math
import re
from typing import Callable, Iterator, List, Sequence, TypeVar

T = TypeVar("T")
EmbeddingFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    """
    Yield fixed-size batches from an in-memory sequence.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def validate_shard(shard_index: int, total_shards: int) -> None:
    """
    Validate distributed worker shard coordinates.
    """

    if total_shards <= 0:
        raise ValueError("total_shards must be greater than zero")

    if shard_index < 0 or shard_index >= total_shards:
        raise ValueError(
            f"shard_index must be between 0 and {total_shards - 1}, "
            f"got {shard_index}"
        )


def shard_filter_sql(id_column: str, shard_index: int, total_shards: int) -> str:
    """
    Build a deterministic BigQuery shard filter.

    BigQuery FARM_FINGERPRINT keeps each VM on a stable slice of the input
    table without requiring expensive OFFSET scans across 50M records.
    """

    validate_shard(shard_index, total_shards)

    if not id_column.replace("_", "").isalnum():
        raise ValueError(f"Unsafe BigQuery column name: {id_column}")

    return (
        f"MOD(ABS(FARM_FINGERPRINT(CAST(`{id_column}` AS STRING))), "
        f"{total_shards}) = {shard_index}"
    )


def split_sentences(text: str) -> List[str]:
    """
    Split clinical text into sentence-like units before semantic chunking.
    """

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """
    Compute cosine similarity without adding a heavy dependency.
    """

    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimension")

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def semantic_chunks(
    text: str,
    embed_sentences: EmbeddingFn,
    similarity_threshold: float = 0.72,
    min_sentences_per_chunk: int = 1,
) -> List[str]:
    """
    Split text by semantic topic shifts instead of fixed word counts.

    Consecutive sentences stay in the same chunk while their embeddings remain
    semantically similar. A new chunk starts when cosine similarity drops below
    the threshold, which is useful for clinical notes that move between topics
    such as medication refills, dosage instructions, and specialist visits.
    """

    sentences = split_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    embeddings = embed_sentences(sentences)

    if len(embeddings) != len(sentences):
        raise ValueError("Embedding function must return one embedding per sentence")

    chunks: List[str] = []
    current_chunk = [sentences[0]]

    for index in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[index - 1], embeddings[index])

        should_start_new_chunk = (
            similarity < similarity_threshold
            and len(current_chunk) >= min_sentences_per_chunk
        )

        if should_start_new_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[index]]
        else:
            current_chunk.append(sentences[index])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
