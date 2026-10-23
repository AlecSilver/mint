#!/usr/bin/env python3
"""
Extract per-residue MINT embeddings for labelled protein pairs, aligned with
the per-residue interface labels from label_interfaces.py.

Input: JSONL, one line per pair (trainset.json):
    {"id": ..., "cluster": ..., "seq1": ..., "seq2": ..., "labels1": [...], "labels2": [...]}

Both chains go through MINT together (chain_ids 0 / 1), so each residue's
embedding is conditioned on its partner. Chains longer than --max-res residues
are cropped to a window (MINT's limit is 512 tokens per chain incl. <cls>/<eos>),
and the labels are cropped with the same window.

Output: --out-dir/shard_00000.pt, shard_00001.pt, ...  Each shard is a dict:
    ids           list[str]            pair ids, in input order
    cluster       int64  [P]
    layers        list[int]            MINT layers saved, in the order of emb's dim 1
    emb           float16 [N, K, 1280] residue embeddings for K = len(layers) layers,
                                       flat over all pairs
                                       (pair p = rows offsets[p]:offsets[p+1],
                                        chain-1 residues first, then chain-2)
    labels        int8   [N]           1 interface / 0 not / -1 unresolved
    chain         int8   [N]           0 = seq1, 1 = seq2
    resid         int32  [N]           0-based residue index in the full seq
    offsets       int64  [P+1]
    crop_start    int32  [P, 2]        window start in seq1 / seq2
    full_len      int32  [P, 2]        uncropped lengths of seq1 / seq2

Shards that already exist are skipped, so an interrupted run can be resumed.
--out-dir/meta.json records the settings that change shard contents (layers, crop, ...);
resuming into a folder made with different settings is refused.

Layer 33 (the final one) has MINT's last layer norm applied; layers 1-32 are raw, so their
scales differ. Saving several layers costs K x the disk/RAM of one.

Usage (in the mint conda env, from interaction_head/):
    python extract_residue_embeddings.py data/trainset.json data/emb_train/ --limit 200   # quick test
    python extract_residue_embeddings.py data/trainset.json data/emb_train/
    python extract_residue_embeddings.py data/trainset.json data/emb_train/ --labeled-only
    python extract_residue_embeddings.py data/trainset.json data/emb_l24-33/ --layers 24 28 33

Loading for training:
    shard = torch.load("data/emb_train/shard_00000.pt")
    keep = shard["labels"] >= 0
    X, y = shard["emb"][keep].float(), shard["labels"][keep]      # X: [n, K, 1280]
    X24 = X[:, shard["layers"].index(24)]                           # one layer: [n, 1280]
"""
import argparse
import itertools
import json
import os
import random
import zlib

import torch
from tqdm import tqdm

from mint.data import Alphabet
from mint.helpers.extract import load_config
from mint.model.esm2 import ESM2

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_pairs(path, limit=None):
    with open(path) as f:
        n = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if len(rec["seq1"]) != len(rec["labels1"]) or len(rec["seq2"]) != len(rec["labels2"]):
                print(f"skipping {rec['id']}: sequence/label length mismatch")
                continue
            yield rec
            n += 1
            if limit and n >= limit:
                return


def crop_window(seq_len, labels, max_res, mode, rng):
    """Return the start of a max_res-long window over a chain of length seq_len."""
    if seq_len <= max_res:
        return 0
    n_starts = seq_len - max_res + 1
    if mode == "start":
        return 0
    if mode == "center":
        return (seq_len - max_res) // 2
    if mode == "random":
        return rng.randrange(n_starts)
    # "interface": window covering the most interface residues (ties -> first)
    hits = [1 if l == 1 else 0 for l in labels]
    best = cur = sum(hits[:max_res])
    best_start = 0
    for s in range(1, n_starts):
        cur += hits[s + max_res - 1] - hits[s - 1]
        if cur > best:
            best, best_start = cur, s
    return best_start


def prepare(rec, max_res, mode):
    # Seed per pair id so crops are reproducible regardless of batching/resume
    rng = random.Random(zlib.crc32(rec["id"].encode()))
    out = {"id": rec["id"], "cluster": rec.get("cluster", -1), "starts": [], "seqs": [], "labels": [],
           "full_len": [len(rec["seq1"]), len(rec["seq2"])]}
    for seq, labels in ((rec["seq1"], rec["labels1"]), (rec["seq2"], rec["labels2"])):
        start = crop_window(len(seq), labels, max_res, mode, rng)
        out["starts"].append(start)
        out["seqs"].append(seq[start:start + max_res])
        out["labels"].append(labels[start:start + max_res])
    return out


def encode_batch(items, alphabet):
    """Same layout as mint's CollateFn: [chain-1 block | chain-2 block], each padded to batch max."""
    blocks, chain_ids, widths = [], [], []
    for c in range(2):
        encoded = [alphabet.encode("<cls>" + it["seqs"][c].replace("J", "L") + "<eos>") for it in items]
        width = max(len(e) for e in encoded)
        tokens = torch.full((len(items), width), alphabet.padding_idx, dtype=torch.int64)
        for i, e in enumerate(encoded):
            tokens[i, :len(e)] = torch.tensor(e, dtype=torch.int64)
        blocks.append(tokens)
        chain_ids.append(torch.full_like(tokens, c, dtype=torch.int32))
        widths.append(width)
    return torch.cat(blocks, -1), torch.cat(chain_ids, -1), widths


def token_batches(items, max_tokens):
    """Group length-sorted items so that batch_size * padded_len <= max_tokens."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["seqs"][0]) + len(items[i]["seqs"][1]))
    batch, longest = [], 0
    for i in order:
        n = len(items[i]["seqs"][0]) + len(items[i]["seqs"][1]) + 4
        if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch


@torch.no_grad()
def embed_shard(items, model, alphabet, device, max_tokens, layers, labeled_only):
    per_item = [None] * len(items)
    for idx in token_batches(items, max_tokens):
        batch = [items[i] for i in idx]
        tokens, chain_ids, widths = encode_batch(batch, alphabet)
        out = model(tokens.to(device), chain_ids.to(device), repr_layers=layers)["representations"]
        reps = torch.stack([out[l] for l in layers], dim=2).to(torch.float16).cpu()  # [B, T, K, D]
        for b, i in enumerate(idx):
            it = items[i]
            parts = []
            for c in range(2):
                L = len(it["seqs"][c])
                first = (widths[0] if c else 0) + 1  # skip <cls>
                emb = reps[b, first:first + L]
                labels = torch.tensor(it["labels"][c], dtype=torch.int8)
                resid = torch.arange(it["starts"][c], it["starts"][c] + L, dtype=torch.int32)
                if labeled_only:
                    keep = labels >= 0
                    emb, labels, resid = emb[keep], labels[keep], resid[keep]
                parts.append((emb, labels, torch.full_like(labels, c), resid))
            per_item[i] = parts

    counts = [sum(len(p[1]) for p in parts) for parts in per_item]
    offsets = torch.zeros(len(items) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int64), 0)
    flat = [p for parts in per_item for p in parts]
    return {
        "ids": [it["id"] for it in items],
        "layers": list(layers),
        "cluster": torch.tensor([it["cluster"] for it in items], dtype=torch.int64),
        "emb": torch.cat([p[0] for p in flat]),
        "labels": torch.cat([p[1] for p in flat]),
        "chain": torch.cat([p[2] for p in flat]),
        "resid": torch.cat([p[3] for p in flat]),
        "offsets": offsets,
        "crop_start": torch.tensor([it["starts"] for it in items], dtype=torch.int32),
        "full_len": torch.tensor([it["full_len"] for it in items], dtype=torch.int32),
    }


def check_meta(out_dir, meta):
    """Write out_dir/meta.json, or refuse if existing shards were made with other settings."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{out_dir} was made with different settings (old, new): {diff}\n"
                             "use a new output folder, or delete it to start over")
    elif any(n.startswith("shard_") for n in os.listdir(out_dir)):
        raise SystemExit(f"{out_dir} has shards from an older version of this script (no meta.json); "
                         "use a new output folder")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="JSONL from label_interfaces.py (e.g. trainset.json)")
    parser.add_argument("out_dir", help="directory for shard_XXXXX.pt files")
    parser.add_argument("--checkpoint", default=os.path.join(REPO_DIR, "mint.ckpt"))
    parser.add_argument("--config", default=os.path.join(REPO_DIR, "data", "esm2_t33_650M_UR50D.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", "--layer", type=int, nargs="+", default=[33],
                        help="layer(s) to extract, e.g. --layers 24 28 33 (33 = final, as in MINTWrapper)")
    parser.add_argument("--max-res", type=int, default=510, help="max residues per chain (510 + <cls>/<eos> = 512)")
    parser.add_argument("--crop", choices=["random", "interface", "center", "start"], default="random",
                        help="window for chains longer than --max-res; 'random' is seeded per pair id")
    parser.add_argument("--max-tokens", type=int, default=4096, help="token budget per forward pass")
    parser.add_argument("--shard-size", type=int, default=200,
                        help="pairs per output shard (each pair holds up to ~2.6 MB of embeddings "
                             "per saved layer in RAM)")
    parser.add_argument("--labeled-only", action="store_true", help="drop residues with label -1")
    parser.add_argument("--fp32", action="store_true", help="run the model in fp32 (default fp16)")
    parser.add_argument("--limit", type=int, help="only process the first N pairs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    layers = sorted(set(args.layers))
    if not all(0 <= l <= cfg.encoder_layers for l in layers):
        parser.error(f"--layers must be between 0 and {cfg.encoder_layers}")
    check_meta(args.out_dir, {"input": os.path.abspath(args.input), "layers": layers,
                              "max_res": args.max_res, "crop": args.crop,
                              "labeled_only": args.labeled_only, "shard_size": args.shard_size})
    # Cast before loading and mmap the checkpoint: MINTWrapper holds the fp32
    # model and the full 3.2 GB checkpoint in memory at once (~6 GB), which
    # overflows both a 6 GB GPU and the WSL VM's RAM
    model = ESM2(num_layers=cfg.encoder_layers, embed_dim=cfg.encoder_embed_dim,
                 attention_heads=cfg.encoder_attention_heads,
                 token_dropout=cfg.token_dropout, use_multimer=True)
    if not args.fp32:
        model.half()
    ckpt = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict({k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()})
    del ckpt
    model.eval().to(args.device)
    alphabet = Alphabet.from_architecture("ESM-1b")

    total = args.limit or sum(1 for _ in open(args.input))
    pairs = read_pairs(args.input, args.limit)
    with tqdm(total=total, unit="pair") as pbar:
        for shard_idx in itertools.count():
            chunk = list(itertools.islice(pairs, args.shard_size))
            if not chunk:
                break
            path = os.path.join(args.out_dir, f"shard_{shard_idx:05d}.pt")
            if not os.path.exists(path):
                items = [prepare(rec, args.max_res, args.crop) for rec in chunk]
                shard = embed_shard(items, model, alphabet, args.device, args.max_tokens,
                                    layers, args.labeled_only)
                torch.save(shard, path + ".tmp")
                os.replace(path + ".tmp", path)
            pbar.update(len(chunk))


if __name__ == "__main__":
    main()
