"""Frozen-workload and vLLM sampling helpers for the serving runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def dataset_root(input_root: Path, dataset: str) -> Path:
    """Accept either a matrix root or one already-selected dataset root."""

    nested = input_root / dataset
    return nested if nested.is_dir() else input_root


def load_metadata(input_root: Path, dataset: str, n: int) -> dict[str, Any]:
    root = dataset_root(input_root, dataset)
    paths = sorted((root / f"n{n}" / "traces").glob("*.json"))
    if len(paths) != 1:
        raise ValueError(f"expected one {dataset} N={n} trace")
    return json.loads(paths[0].read_text(encoding="utf-8"))["metadata"]


def token_hash(token_ids: list[int]) -> str:
    payload = ",".join(map(str, token_ids))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def sampling_params(metadata: dict[str, Any], seed: int, params_type: Any) -> Any:
    contract = metadata["sampling_params"]
    from vllm.sampling_params import RequestOutputKind

    return params_type(
        n=1,
        temperature=float(contract["temperature"]),
        top_p=float(contract["top_p"]),
        top_k=int(contract["top_k"]),
        min_p=float(contract["min_p"]),
        presence_penalty=float(contract["presence_penalty"]),
        repetition_penalty=float(contract["repetition_penalty"]),
        max_tokens=int(metadata["max_tokens"]),
        ignore_eos=bool(metadata["ignore_eos"]),
        seed=seed,
        detokenize=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def replay_sampling_params(
    metadata: dict[str, Any], seed: int, tokens: int, params_type: Any
) -> Any:
    contract = metadata["sampling_params"]
    from vllm.sampling_params import RequestOutputKind

    return params_type(
        n=1,
        temperature=float(contract["temperature"]),
        top_p=float(contract["top_p"]),
        top_k=int(contract["top_k"]),
        min_p=float(contract["min_p"]),
        presence_penalty=float(contract["presence_penalty"]),
        repetition_penalty=float(contract["repetition_penalty"]),
        max_tokens=tokens,
        min_tokens=tokens,
        ignore_eos=True,
        seed=seed,
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def load_quality_traces(
    specifications: list[str],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for specification in specifications:
        dataset, separator, raw_path = specification.partition("=")
        if not separator or not dataset or not raw_path:
            raise ValueError(f"invalid quality trace: {specification}")
        path = Path(raw_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (dataset, str(row["parent_id"]), int(row["candidate_index"]))
            if key in rows:
                raise ValueError(f"duplicate quality candidate: {key}")
            rows[key] = row
    return rows


__all__ = [
    "dataset_root",
    "load_metadata",
    "load_quality_traces",
    "replay_sampling_params",
    "sampling_params",
    "token_hash",
]
