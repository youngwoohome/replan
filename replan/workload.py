"""Load full-N traces together with prompts and reference answers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateSpec:
    index: int
    seed: int
    target_tokens: int
    token_ids_sha256: str


@dataclass(frozen=True)
class QualityParent:
    parent_id: str
    dataset: str
    prompt: str
    prompt_token_ids: tuple[int, ...] | None
    reference_answer: str
    candidates: tuple[CandidateSpec, ...]

    @property
    def prompt_input(self) -> str | dict[str, list[int]]:
        """Use frozen token IDs when a collected trace provides them."""

        if self.prompt_token_ids is None:
            return self.prompt
        return {"prompt_token_ids": list(self.prompt_token_ids)}


def _prompt_rows(trace: dict[str, Any], prompt_root: Path) -> list[dict[str, Any]]:
    metadata = trace.get("metadata") or {}
    configured = metadata.get("prompt_jsonl")
    candidates: list[Path] = []
    if isinstance(configured, str):
        candidates.append(prompt_root / Path(configured).name)
    dataset = str(metadata.get("dataset", ""))
    candidates.extend(
        [
            prompt_root / f"{dataset}_all.jsonl",
            prompt_root / f"{dataset}_80.jsonl",
        ]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"no prompt JSONL found under {prompt_root}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_quality_workload(
    trace_root: Path,
    prompt_root: Path,
    *,
    max_parents: int | None = None,
    parent_offset: int = 0,
) -> list[QualityParent]:
    if parent_offset < 0:
        raise ValueError("parent_offset must be non-negative")
    trace_paths = sorted(trace_root.glob("*.json"))
    if len(trace_paths) != 1:
        raise ValueError(f"expected one trace JSON, found {len(trace_paths)}")
    trace = json.loads(trace_paths[0].read_text(encoding="utf-8"))
    if trace.get("format") != "tail_blend_length_replay_v1":
        raise ValueError("unsupported trace format")
    if trace.get("complete") is not True:
        raise ValueError("trace is incomplete")
    metadata = trace.get("metadata") or {}
    dataset = str(metadata["dataset"])
    expected_n = int(metadata["generated_bon"])
    prompts = _prompt_rows(trace, prompt_root)
    parents: list[QualityParent] = []
    for request in trace.get("requests", []):
        request_index = int(request["request_index"])
        prompt = prompts[request_index]
        reference = prompt.get("reference_answer")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"missing reference answer for request {request_index}")
        raw_prompt_token_ids = prompt.get("prompt_token_ids")
        prompt_token_ids: tuple[int, ...] | None = None
        prompt_text = str(prompt.get("prompt", ""))
        if isinstance(raw_prompt_token_ids, list):
            prompt_token_ids = tuple(int(token) for token in raw_prompt_token_ids)
            prompt_hash = hashlib.sha256(
                ",".join(map(str, prompt_token_ids)).encode("ascii")
            ).hexdigest()
            if prompt_hash != request.get("prompt_token_ids_sha256"):
                raise ValueError(
                    f"prompt token hash mismatch for request {request_index}"
                )
        else:
            prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if prompt_hash != request.get("prompt_sha256"):
                raise ValueError(f"prompt hash mismatch for request {request_index}")
        branches = tuple(
            CandidateSpec(
                index=int(branch["index"]),
                seed=int(branch["seed"]),
                target_tokens=int(branch["target_tokens"]),
                token_ids_sha256=str(branch["token_ids_sha256"]),
            )
            for branch in sorted(
                request["branches"], key=lambda value: int(value["index"])
            )
        )
        if len(branches) != expected_n:
            raise ValueError(
                f"expected N={expected_n}, got {len(branches)} for {request_index}"
            )
        parents.append(
            QualityParent(
                parent_id=f"{dataset}-g{request_index}",
                dataset=dataset,
                prompt=prompt_text,
                prompt_token_ids=prompt_token_ids,
                reference_answer=reference,
                candidates=branches,
            )
        )
    selected = parents[parent_offset:]
    if max_parents is not None:
        selected = selected[:max_parents]
    if not selected:
        raise ValueError("no parents selected")
    return selected


__all__ = ["CandidateSpec", "QualityParent", "load_quality_workload"]
