#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_PATTERN = re.compile(
    r"^(Contiguous|Offset \d+|Memory stride \d+|Gather pattern \d+): "
    r"([0-9.eE+-]+) ms, ([0-9.eE+-]+) GB/s$"
)


def run_benchmark(executable, problem_size, block_size, offset, stride, gather, warmups,
                  repetitions):
    command = [
        str(executable),
        str(problem_size),
        str(block_size),
        str(offset),
        str(stride),
        str(gather),
        str(warmups),
        str(repetitions),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)

    results = {}
    for line in completed.stdout.splitlines():
        match = RESULT_PATTERN.match(line)
        if match:
            results[match.group(1)] = {
                "average_ms": float(match.group(2)),
                "bandwidth_gbps": float(match.group(3)),
            }

    if len(results) != 4:
        raise RuntimeError(f"Could not parse benchmark output:\n{completed.stdout}")

    return results


def plot_results(rows, metric, ylabel, output_path):
    categories = [
        ("offset", "Offset"),
        ("stride", "Memory stride"),
        ("gather", "Gather pattern"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for axis, (category, title) in zip(axes, categories):
        selected = [row for row in rows if row["category"] == category]
        axis.plot(
            [row["value"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=2,
        )
        axis.set_title(title)
        axis.set_xlabel("Pattern value")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sweep Lab 03 memory-access patterns")
    parser.add_argument("executable", type=Path)
    parser.add_argument("--problem-size", type=int, default=10_000_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("./lab03-results"))
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"Executable does not exist: {executable}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for offset in [0, 1, 8]:
        result = run_benchmark(executable, args.problem_size, args.block_size, offset, 1, 0,
                               args.warmups, args.repetitions)[f"Offset {offset}"]
        rows.append({"category": "offset", "value": offset, **result})

    for stride in [1, 2, 8, 32]:
        result = run_benchmark(executable, args.problem_size, args.block_size, 0, stride, 0,
                               args.warmups, args.repetitions)[f"Memory stride {stride}"]
        rows.append({"category": "stride", "value": stride, **result})

    for gather in [0, 1, 2]:
        result = run_benchmark(executable, args.problem_size, args.block_size, 0, 1, gather,
                               args.warmups, args.repetitions)[f"Gather pattern {gather}"]
        rows.append({"category": "gather", "value": gather, **result})

    csv_path = args.output_dir / "memory-coalescing-sweep.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows, "bandwidth_gbps", "Effective bandwidth (GB/s)",
                 args.output_dir / "bandwidth.png")
    plot_results(rows, "average_ms", "Average latency (ms)",
                 args.output_dir / "latency.png")

    for row in rows:
        print(
            f"{row['category']:>6} {row['value']:>2}: "
            f"{row['average_ms']:.6f} ms, {row['bandwidth_gbps']:.3f} GB/s"
        )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
