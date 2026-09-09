import hashlib
import json

from replan.workload import load_quality_workload


def _trace(prompt_hash_field: str, prompt_hash: str) -> dict:
    return {
        "format": "tail_blend_length_replay_v1",
        "complete": True,
        "metadata": {
            "dataset": "math",
            "generated_bon": 1,
            "prompt_jsonl": "math_all.jsonl",
        },
        "requests": [
            {
                "request_index": 0,
                prompt_hash_field: prompt_hash,
                "branches": [
                    {
                        "index": 0,
                        "seed": 7,
                        "target_tokens": 4,
                        "token_ids_sha256": "candidate-hash",
                    }
                ],
            }
        ],
    }


def _write_workload(tmp_path, prompt: dict, trace: dict):
    trace_root = tmp_path / "traces"
    prompt_root = tmp_path / "prompts"
    trace_root.mkdir()
    prompt_root.mkdir()
    (trace_root / "math.json").write_text(json.dumps(trace), encoding="utf-8")
    (prompt_root / "math_all.jsonl").write_text(
        json.dumps(prompt) + "\n", encoding="utf-8"
    )
    return load_quality_workload(trace_root, prompt_root)[0]


def test_text_prompt_is_hash_checked(tmp_path) -> None:
    text = "What is 1 + 1?"
    digest = hashlib.sha256(text.encode()).hexdigest()
    parent = _write_workload(
        tmp_path,
        {"prompt": text, "reference_answer": "2"},
        _trace("prompt_sha256", digest),
    )

    assert parent.prompt_input == text


def test_frozen_prompt_tokens_are_hash_checked(tmp_path) -> None:
    token_ids = [10, 20, 30]
    digest = hashlib.sha256(b"10,20,30").hexdigest()
    parent = _write_workload(
        tmp_path,
        {"prompt_token_ids": token_ids, "reference_answer": "2"},
        _trace("prompt_token_ids_sha256", digest),
    )

    assert parent.prompt_input == {"prompt_token_ids": token_ids}
