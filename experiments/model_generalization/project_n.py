"""Project a completed fresh N-max trace to a smaller prefix N."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--source-n", type=int, required=True)
    parser.add_argument("--target-n", type=int, required=True)
    args = parser.parse_args()
    if not 0 < args.target_n < args.source_n:
        raise ValueError("target N must be positive and smaller than source N")
    source_paths = sorted(
        (args.input_root / f"n{args.source_n}" / "traces").glob("*.json")
    )
    if len(source_paths) != 1:
        raise ValueError(f"expected one source trace, found {len(source_paths)}")
    source = json.loads(source_paths[0].read_text(encoding="utf-8"))
    if source.get("complete") is not True:
        raise ValueError("source trace is incomplete")
    target = dict(source)
    target["metadata"] = dict(source["metadata"])
    target["metadata"]["generated_bon"] = args.target_n
    target["metadata"]["projected_from_n"] = args.source_n
    target["requests"] = []
    for request in source["requests"]:
        projected = dict(request)
        projected["branches"] = list(request["branches"][: args.target_n])
        if len(projected["branches"]) != args.target_n:
            raise ValueError("source request has too few branches")
        target["requests"].append(projected)
    output_dir = args.input_root / f"n{args.target_n}" / "traces"
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / source_paths[0].name.replace(
        f"n{args.source_n}", f"n{args.target_n}"
    )
    output.write_text(json.dumps(target, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
