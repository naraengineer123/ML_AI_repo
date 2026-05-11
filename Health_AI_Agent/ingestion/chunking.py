from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


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
