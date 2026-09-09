"""Regenerate full-N candidates and preserve answer-quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from replan.answers import extract_answer, numeric_answer_is_correct
from replan.workload import load_quality_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--prompt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-parents", type=int)
    parser.add_argument("--parent-offset", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--trace-consistency",
        choices=("strict", "record"),
        default="strict",
        help=(
            "strict rejects outputs that differ from the source token trace; "
            "record keeps the newly generated output and records every mismatch"
        ),
    )
    return parser.parse_args()


def _token_hash(token_ids: list[int]) -> str:
    payload = ",".join(map(str, token_ids))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    parents = load_quality_workload(
        args.trace_root,
        args.prompt_root,
        max_parents=args.max_parents,
        parent_offset=args.parent_offset,
    )
    trace_paths = sorted(args.trace_root.glob("*.json"))
    trace = json.loads(trace_paths[0].read_text(encoding="utf-8"))
    trace_metadata = trace["metadata"]
    sampling_contract = trace_metadata["sampling_params"]
    args.output_dir.mkdir(parents=True)
    candidate_path = args.output_dir / "candidates.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "format": "replan_candidates_v1",
        "status": "running",
        "dataset": parents[0].dataset,
        "parents": len(parents),
        "n": len(parents[0].candidates),
        "trace_root": str(args.trace_root.resolve()),
        "prompt_root": str(args.prompt_root.resolve()),
        "model": args.model,
        "completed_parents": 0,
        "hash_mismatches": 0,
        "trace_consistency": args.trace_consistency,
        "replay_length_source": "actual_tokens",
        "source_sampling_contract": sampling_contract,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

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
            enable_prefix_caching=args.enable_prefix_caching,
            enable_chunked_prefill=True,
            enforce_eager=args.enforce_eager,
            disable_log_stats=False,
        ),
        enable_multiprocessing=False,
    )

    mismatches = 0
    with candidate_path.open("w", encoding="utf-8") as output_handle:
        for parent_number, parent in enumerate(parents, start=1):
            mismatches_before_parent = mismatches
            # Reproduce the trace generator's logical request exactly. vLLM's
            # n-way sampling stream is not equivalent to N independent
            # requests, even when the apparent per-branch seeds are supplied.
            engine.add_request(
                parent.parent_id,
                parent.prompt,
                SamplingParams(
                    n=len(parent.candidates),
                    temperature=float(sampling_contract["temperature"]),
                    top_p=float(sampling_contract["top_p"]),
                    top_k=int(sampling_contract["top_k"]),
                    min_p=float(sampling_contract["min_p"]),
                    presence_penalty=float(
                        sampling_contract["presence_penalty"]
                    ),
                    repetition_penalty=float(
                        sampling_contract["repetition_penalty"]
                    ),
                    max_tokens=int(trace_metadata["max_tokens"]),
                    ignore_eos=bool(trace_metadata["ignore_eos"]),
                    seed=parent.candidates[0].seed,
                    detokenize=True,
                    output_kind=RequestOutputKind.FINAL_ONLY,
                ),
            )
            final_output = None
            while engine.has_unfinished_requests():
                for request_output in engine.step():
                    if request_output.request_id == parent.parent_id and request_output.finished:
                        final_output = request_output
            if final_output is None:
                raise RuntimeError(f"missing final output for {parent.parent_id}")
            samples = {
                int(sample.index): sample for sample in final_output.outputs
            }
            if len(samples) != len(parent.candidates):
                raise RuntimeError(
                    f"expected {len(parent.candidates)} samples, got {len(samples)}"
                )
            completed: list[dict[str, Any]] = []
            for candidate in parent.candidates:
                sample = samples[candidate.index]
                actual_hash = _token_hash(sample.token_ids)
                hash_matches = actual_hash == candidate.token_ids_sha256
                mismatches += int(not hash_matches)
                answer = extract_answer(sample.text)
                completed.append(
                    {
                        "format": "replan_candidate_v1",
                        "parent_id": parent.parent_id,
                        "dataset": parent.dataset,
                        "reference_answer": parent.reference_answer,
                        "candidate_index": candidate.index,
                        "seed": candidate.seed,
                        "target_tokens": candidate.target_tokens,
                        "actual_tokens": len(sample.token_ids),
                        "expected_token_ids_sha256": candidate.token_ids_sha256,
                        "actual_token_ids_sha256": actual_hash,
                        "hash_matches_trace": hash_matches,
                        "extracted_answer": answer,
                        "correct": numeric_answer_is_correct(
                            answer, parent.reference_answer
                        ),
                        "text": sample.text,
                    }
                )
            for row in completed:
                output_handle.write(json.dumps(row) + "\n")
            output_handle.flush()
            manifest["completed_parents"] = parent_number
            manifest["hash_mismatches"] = mismatches
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            parent_mismatches = mismatches - mismatches_before_parent
            if parent_mismatches and args.trace_consistency == "strict":
                manifest["status"] = "invalid"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                raise RuntimeError(
                    f"{parent.parent_id} has {parent_mismatches} token-hash "
                    "mismatches; refusing to collect later parents"
                )

    manifest["status"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
