"""Collect model-specific BoN candidates and a matching replay workload.

This utility is deliberately separate from ``replan.collect``.  The
paper collector replays an existing model trace, whereas model generalization
requires new chat-template tokens, output lengths, and answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from replan.answers import extract_answer, numeric_answer_is_correct

CHATML_MESSAGE = re.compile(
    r"<\|im_start\|>(system|user)\n(.*?)<\|im_end\|>", re.DOTALL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--source-prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--parents", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the last fully written parent in an existing output root",
    )
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _token_hash(token_ids: list[int]) -> str:
    return _sha256_bytes(",".join(map(str, token_ids)).encode("ascii"))


def _prompt_token_hash(token_ids: list[int]) -> str:
    return _token_hash(token_ids)


def _messages(prompt: str) -> list[dict[str, str]]:
    messages = [
        {"role": role, "content": content.strip()}
        for role, content in CHATML_MESSAGE.findall(prompt)
    ]
    if not messages or not any(row["role"] == "user" for row in messages):
        raise ValueError("source prompt is not the expected system/user ChatML")
    for row in messages:
        if row["role"] == "system" and row["content"].startswith(
            "You are Qwen, created by Alibaba Cloud."
        ):
            row["content"] = "You are a helpful assistant."
    return messages


def _chat_tokens(tokenizer: Any, messages: list[dict[str, str]], thinking: bool | None) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if thinking is not None:
        kwargs["enable_thinking"] = thinking
    encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_exists = args.output_root.exists()
    if output_exists and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    source_trace = json.loads(args.source_trace.read_text(encoding="utf-8"))
    source_metadata = source_trace["metadata"]
    source_rows = [
        json.loads(line)
        for line in args.source_prompts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.parents]
    if len(source_rows) != args.parents:
        raise ValueError(f"requested {args.parents} parents, found {len(source_rows)}")

    prompts_dir = args.output_root / "prompts"
    traces_dir = args.output_root / f"n{args.n}" / "traces"
    quality_dir = args.output_root / "quality"
    for path in (prompts_dir, traces_dir, quality_dir):
        path.mkdir(parents=True, exist_ok=args.resume)
    prompt_path = prompts_dir / f"{source_metadata['dataset']}_all.jsonl"
    candidate_path = quality_dir / "candidates.jsonl"
    manifest_path = args.output_root / "manifest.json"
    trace_path = traces_dir / f"{source_metadata['dataset']}_n{args.n}.json"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prepared: list[tuple[dict[str, Any], list[int]]] = []
    with prompt_path.open("w", encoding="utf-8") as handle:
        for row in source_rows:
            token_ids = _chat_tokens(
                tokenizer, _messages(str(row["prompt"])), args.enable_thinking
            )
            output_row = dict(row)
            output_row["source_prompt"] = output_row.pop("prompt")
            output_row["prompt"] = ""
            output_row["prompt_token_ids"] = token_ids
            output_row["prompt_token_ids_sha256"] = _prompt_token_hash(token_ids)
            handle.write(json.dumps(output_row) + "\n")
            prepared.append((output_row, token_ids))

    sampling = dict(source_metadata["sampling_params"])
    max_tokens = args.max_tokens or int(source_metadata["max_tokens"])
    metadata = dict(source_metadata)
    metadata.update(
        {
            "generated_bon": args.n,
            "num_requests": args.parents,
            "model": args.model_label,
            "runtime_model_path": args.model,
            "tokenizer": args.model_label,
            "max_tokens": max_tokens,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "prompt_jsonl": str(prompt_path),
            "prompt_format": "pretokenized_model_chat_template_v1",
            "enable_thinking": args.enable_thinking,
            "source_trace": str(args.source_trace),
        }
    )
    if output_exists:
        if not trace_path.is_file() or not manifest_path.is_file():
            raise ValueError("resume output is missing its trace or manifest")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = (
            manifest.get("format") == "replan_model_generalization_collection_v1"
            and manifest.get("model") == args.model_label
            and manifest.get("dataset") == source_metadata["dataset"]
            and int(manifest.get("n", -1)) == args.n
        )
        if not expected:
            raise ValueError("resume configuration does not match existing output")
        completed_parents = int(manifest["completed_parents"])
        if completed_parents > args.parents:
            raise ValueError("existing output contains more parents than requested")
        if len(trace.get("requests", [])) != completed_parents:
            raise ValueError("trace checkpoint is inconsistent with its manifest")
        candidate_lines = sum(
            bool(line.strip())
            for line in candidate_path.read_text(encoding="utf-8").splitlines()
        )
        if candidate_lines != completed_parents * args.n:
            raise ValueError("candidate checkpoint is inconsistent with its manifest")
        trace["complete"] = False
        trace["metadata"]["num_requests"] = args.parents
        manifest["status"] = "running"
        manifest["parents"] = args.parents
        manifest["resumed_unix_s"] = time.time()
    else:
        completed_parents = 0
        trace = {
            "format": "tail_blend_length_replay_v1",
            "complete": False,
            "metadata": metadata,
            "requests": [],
        }
        manifest = {
            "format": "replan_model_generalization_collection_v1",
            "status": "running",
            "model": args.model_label,
            "dataset": source_metadata["dataset"],
            "parents": args.parents,
            "n": args.n,
            "completed_parents": 0,
            "started_unix_s": time.time(),
        }
    _write_json(trace_path, trace)
    _write_json(manifest_path, manifest)
    if completed_parents == args.parents:
        trace["complete"] = True
        manifest["status"] = "completed"
        _write_json(trace_path, trace)
        _write_json(manifest_path, manifest)
        return

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLMEngine, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import RequestOutputKind

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=args.enforce_eager,
            disable_log_stats=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
        ),
        enable_multiprocessing=False,
    )

    base_seed = int(source_metadata["base_sampling_seed"])
    stride = int(source_metadata["sampling_seed_stride"])
    candidate_mode = "a" if completed_parents else "w"
    with candidate_path.open(candidate_mode, encoding="utf-8") as candidate_handle:
        for request_index in range(completed_parents, len(prepared)):
            prompt_row, prompt_token_ids = prepared[request_index]
            parent_id = f"{source_metadata['dataset']}-g{request_index}"
            request_seed = base_seed + request_index * stride
            engine.add_request(
                parent_id,
                {"prompt_token_ids": prompt_token_ids},
                SamplingParams(
                    n=args.n,
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    min_p=float(sampling["min_p"]),
                    presence_penalty=float(sampling["presence_penalty"]),
                    repetition_penalty=float(sampling["repetition_penalty"]),
                    max_tokens=max_tokens,
                    ignore_eos=False,
                    seed=request_seed,
                    detokenize=True,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                ),
            )
            final_output = None
            while engine.has_unfinished_requests():
                for request_output in engine.step():
                    if request_output.request_id == parent_id and request_output.finished:
                        final_output = request_output
            if final_output is None:
                raise RuntimeError(f"missing final output for {parent_id}")
            outputs = {int(sample.index): sample for sample in final_output.outputs}
            if len(outputs) != args.n:
                raise RuntimeError(f"expected {args.n} samples, got {len(outputs)}")
            branches: list[dict[str, Any]] = []
            for index in range(args.n):
                sample = outputs[index]
                answer = extract_answer(sample.text)
                token_hash = _token_hash(sample.token_ids)
                candidate_handle.write(
                    json.dumps(
                        {
                            "format": "replan_candidate_v1",
                            "parent_id": parent_id,
                            "dataset": source_metadata["dataset"],
                            "reference_answer": prompt_row["reference_answer"],
                            "candidate_index": index,
                            "seed": request_seed + index,
                            "target_tokens": len(sample.token_ids),
                            "actual_tokens": len(sample.token_ids),
                            "actual_token_ids_sha256": token_hash,
                            "extracted_answer": answer,
                            "correct": numeric_answer_is_correct(
                                answer, str(prompt_row["reference_answer"])
                            ),
                            "finish_reason": sample.finish_reason,
                            "stop_reason": sample.stop_reason,
                            "text": sample.text,
                        }
                    )
                    + "\n"
                )
                branches.append(
                    {
                        "index": index,
                        "seed": request_seed + index,
                        "target_tokens": len(sample.token_ids),
                        "token_ids_sha256": token_hash,
                        "finish_reason": sample.finish_reason,
                        "stop_reason": sample.stop_reason,
                    }
                )
            candidate_handle.flush()
            trace["requests"].append(
                {
                    "request_index": request_index,
                    "prompt_token_ids_sha256": prompt_row[
                        "prompt_token_ids_sha256"
                    ],
                    "branches": branches,
                }
            )
            manifest["completed_parents"] = request_index + 1
            _write_json(trace_path, trace)
            _write_json(manifest_path, manifest)

    trace["complete"] = True
    manifest["status"] = "completed"
    manifest["finished_unix_s"] = time.time()
    _write_json(trace_path, trace)
    _write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
