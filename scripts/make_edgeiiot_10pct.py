"""Create the deterministic 10% Edge-IIoTset subset used by the manuscript.

Input: official DNN-EdgeIIoT-dataset.csv
Output: DNN-EdgeIIoT-10pct-seed42.csv
Sampling is exact within each Attack_type using a two-pass streaming algorithm,
so the source CSV does not need to fit in memory.
"""
import argparse
import csv
import random
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", type=Path, default=Path("DNN-EdgeIIoT-10pct-seed42.csv"))
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-col", default="Attack_type")
    args = ap.parse_args()
    if not math.isfinite(args.fraction) or not 0 < args.fraction <= 1:
        ap.error('--fraction must be finite and in (0, 1]')
    if args.input.resolve() == args.output.resolve():
        ap.error('input and output must be different files')
    def sha256(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    source_hash = sha256(args.input)

    counts = Counter()
    with args.input.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if args.label_col not in (reader.fieldnames or []):
            raise ValueError(f"{args.label_col!r} not found. Columns: {reader.fieldnames}")
        for row in reader:
            counts[row[args.label_col]] += 1

    targets = {cls: max(1, round(n * args.fraction)) for cls, n in counts.items()}
    if not counts:
        raise ValueError('Input has no data rows')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    remaining = counts.copy()
    needed = targets.copy()
    selected = Counter()
    rng = random.Random(args.seed)

    with args.input.open("r", encoding="utf-8", errors="ignore", newline="") as fin, \
         args.output.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            cls = row[args.label_col]
            if needed[cls] > 0:
                p = needed[cls] / remaining[cls]
                if rng.random() < p:
                    writer.writerow(row)
                    needed[cls] -= 1
                    selected[cls] += 1
            remaining[cls] -= 1

    print(f"Saved {sum(selected.values()):,} rows to {args.output}")
    if dict(selected) != targets:
        raise RuntimeError(f'Sampling target mismatch: {dict(selected)} != {targets}')
    if sha256(args.input) != source_hash:
        raise RuntimeError('Input changed during sampling; discard the output')
    manifest = {
        'source_file': args.input.name, 'source_sha256': source_hash,
        'output_file': args.output.name, 'output_sha256': sha256(args.output),
        'fraction': args.fraction, 'seed': args.seed, 'label_column': args.label_col,
        'source_rows': sum(counts.values()), 'output_rows': sum(selected.values()),
        'source_counts': dict(counts), 'target_counts': targets, 'selected_counts': dict(selected),
        'rounding': 'max(1, Python round(class_count * fraction)); ties to even',
        'ordering': 'Source row order affects sample membership; source hash must match',
    }
    args.output.with_suffix(args.output.suffix + '.manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    for cls in sorted(selected):
        print(f"{cls}: {selected[cls]:,} / {counts[cls]:,}")


if __name__ == "__main__":
    main()
