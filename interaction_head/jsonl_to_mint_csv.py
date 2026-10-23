#!/usr/bin/env python3
"""
Convert interface-label JSONL (output of label_interfaces.py) into a CSV that
mint.helpers.extract.CSVDataset can read.

Input: JSONL, one line per pair:
    {"id": ..., "cluster": ..., "seq1": ..., "seq2": ..., "labels1": [...], "labels2": [...]}

Output: CSV with columns  id,cluster,seq1,seq2  (plus labels1,labels2 as JSON
strings with --keep-labels). Row order matches the input, so row i of the CSV
is embedding i from MINT.

Usage (from interaction_head/):
    python jsonl_to_mint_csv.py data/trainset.json data/trainset.csv
    python jsonl_to_mint_csv.py data/trainset.json data/trainset.csv --keep-labels

Then:
    dataset = CSVDataset('data/trainset.csv', 'seq1', 'seq2')
"""
import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="JSONL file from label_interfaces.py")
    parser.add_argument("output", help="CSV file to write")
    parser.add_argument("--keep-labels", action="store_true",
                        help="also write labels1/labels2 columns (JSON-encoded lists)")
    args = parser.parse_args()

    fields = ["id", "cluster", "seq1", "seq2"]
    if args.keep_labels:
        fields += ["labels1", "labels2"]

    n_written = n_skipped = 0
    with open(args.input) as fin, open(args.output, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not rec.get("seq1") or not rec.get("seq2"):
                print(f"line {line_no}: skipping {rec.get('id')} (missing sequence)")
                n_skipped += 1
                continue
            row = {k: rec[k] for k in ("id", "cluster", "seq1", "seq2")}
            if args.keep_labels:
                row["labels1"] = json.dumps(rec["labels1"], separators=(",", ":"))
                row["labels2"] = json.dumps(rec["labels2"], separators=(",", ":"))
            writer.writerow(row)
            n_written += 1

    print(f"Wrote {n_written} pairs to {args.output} ({n_skipped} skipped)")


if __name__ == "__main__":
    main()
