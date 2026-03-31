#!/usr/bin/env python3
"""Extract and summarize benchmark results from aiperf JSON exports."""

import json
import sys
from pathlib import Path


def extract(result_dir: str):
    result_path = Path(result_dir)
    json_files = sorted(result_path.glob("**/profile_export_aiperf.json"))

    if not json_files:
        print(f"No results found in {result_dir}")
        return

    print(f"{'Directory':<60} {'TPS':>10} {'RPS':>10} {'ISL':>8} {'OSL':>8}")
    print("-" * 100)

    for f in json_files:
        d = json.load(open(f))
        tps = d["output_token_throughput"]["avg"]
        rps = d["request_throughput"]["avg"]
        isl = d["input_sequence_length"]["avg"]
        osl = d["output_sequence_length"]["avg"]
        name = str(f.parent.relative_to(result_path))
        print(f"{name:<60} {tps:>10.1f} {rps:>10.2f} {isl:>8.0f} {osl:>8.0f}")


if __name__ == "__main__":
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    extract(result_dir)
