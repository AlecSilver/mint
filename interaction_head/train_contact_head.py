#!/usr/bin/env python3
"""
Train an inter-chain residue contact head on saved MINT embeddings.

Inputs (train and val each):
  * embedding shards from extract_residue_embeddings.py, run on the label_contacts.py JSONL
    (or on a label_interfaces.py JSONL made from the same pairs and --seq-dir)
  * the label_contacts.py JSONL for the same pairs, joined by pair id

Head (TEIM-style, as in downstream/TCR-Epitope/tcr-interface/train_res.py, but residue
features are concatenated instead of multiplied; --pair multiply gives TEIM's version):

    emb1 [L1, K, 1280], emb2 [L2, K, 1280]    K saved layers (--layers); K > 1 -> learned mix
      -> LayerNorm per layer, softmax-weighted layer mix
      -> Linear(1280 -> h), separate for chain 1 / chain 2      a_i, b_j
      -> z_ij = ReLU(LN(a_i + b_j))       --pair add (default);  a_i * b_j with --pair multiply
      -> --blocks residual blocks  z + Conv3x3(ReLU(LN(z))), dilation 1, 2, 4, 8, 1, ...
      -> Conv3x3(h -> 2): contact logit, distance (/ cutoff)

ReLU(LN(a_i + b_j)) is exactly ReLU(LN(W [x_i ; y_j] + c)), a linear layer on the concatenated
residue pair, without materialising the [L1, L2, 2560] map (1.3 GB per 510 x 510 pair in fp16).
Memory then scales with h: about 0.3-0.7 GB per 256 x 256 training crop at h = 128 with AMP.

Training crops each chain to --crop residues (a window around a random contact with
probability 0.8) and randomly swaps chain order, so the head doesn't learn which chain is
listed first. Validation runs on the full saved windows (<= 510 per chain).

Loss: masked BCE on contacts (+ --dist-weight x masked MSE on distance / cutoff, where pairs
beyond the stored cutoff count as the cutoff). Cells with an unresolved / unembedded residue
are ignored.
Metrics: average precision over all val cells, and per-pair precision of the top 10, L/10
and L/5 predictions (L = shorter chain), the usual inter-chain contact metrics.

Usage (any env with torch >= 2.1, e.g. Colab):
    python train_contact_head.py \\
        --train-emb data/emb_train --train-contacts data/trainset_contacts.json \\
        --val-emb data/emb_val --val-contacts data/validset_contacts.json \\
        --out runs/add_h128
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------- data
def read_contacts(path):
    with open(path) as f:
        return {r["id"]: r for r in map(json.loads, filter(str.strip, f))}


def dense_labels(rec):
    """(label, dist) for a label_contacts.py record: label 1/0/-1, dist in Å (inf beyond cutoff)."""
    r1 = np.asarray(rec["resolved1"], dtype=bool)
    r2 = np.asarray(rec["resolved2"], dtype=bool)
    label = np.full((len(r1), len(r2)), -1, dtype=np.int8)
    label[np.ix_(r1, r2)] = 0
    dist = np.full(label.shape, np.inf, dtype=np.float32)
    if rec["contacts"]:
        c = np.asarray(rec["contacts"])
        i, j = c[:, 0].astype(int), c[:, 1].astype(int)
        label[i, j] = 1
        dist[i, j] = c[:, 2]
    return label, dist


class PairData(Dataset):
    """One item per embedded pair that also has contact labels."""

    def __init__(self, emb_dir, contacts_path, layers=None, crop=None, train=False):
        self.contacts = read_contacts(contacts_path)
        self.paths = sorted(glob.glob(os.path.join(emb_dir, "shard_*.pt")))
        if not self.paths:
            raise SystemExit(f"no shards in {emb_dir}")
        self.crop, self.train = crop, train
        self._shards = {}  # opened lazily, per worker

        self.index, n_missing, n_len = [], 0, 0
        for s, path in enumerate(self.paths):
            shard = self.shard(s)
            for p, pid in enumerate(shard["ids"]):
                rec = self.contacts.get(pid)
                if rec is None:
                    n_missing += 1
                elif shard["full_len"][p].tolist() != [len(rec["seq1"]), len(rec["seq2"])]:
                    n_len += 1  # embeddings were made from different sequences
                else:
                    self.index.append((s, p, rec.get("cluster", -1)))
        if n_missing or n_len:
            print(f"{emb_dir}: skipped {n_missing} pairs without contact labels, "
                  f"{n_len} with mismatched sequence lengths")

        saved = self.shard(0)["layers"]
        self.layers = layers or saved
        if any(l not in saved for l in self.layers):
            raise SystemExit(f"--layers {self.layers} not all in the saved layers {saved}")
        self.layer_idx = [saved.index(l) for l in self.layers]
        self.dim = self.shard(0)["emb"].shape[-1]

    def shard(self, s):
        if s not in self._shards:
            # mmap: only the rows of the pairs actually read are paged in
            self._shards[s] = torch.load(self.paths[s], map_location="cpu", mmap=True, weights_only=False)
        return self._shards[s]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        s, p, _ = self.index[k]
        shard = self.shard(s)
        o0, o1 = shard["offsets"][p].item(), shard["offsets"][p + 1].item()
        emb = shard["emb"][o0:o1][:, self.layer_idx]
        chain, resid = shard["chain"][o0:o1].long(), shard["resid"][o0:o1].long()
        label, dist = dense_labels(self.contacts[shard["ids"][p]])

        # scatter each chain's rows into its saved window (rows may be missing with --labeled-only)
        chains, starts = [], []
        for c in range(2):
            sel = chain == c
            start = shard["crop_start"][p, c].item()
            length = (resid[sel].max().item() + 1 - start) if sel.any() else 0
            e = torch.zeros(length, len(self.layer_idx), self.dim, dtype=emb.dtype)
            m = torch.zeros(length, dtype=torch.bool)
            e[resid[sel] - start], m[resid[sel] - start] = emb[sel], True
            chains.append((e, m))
            starts.append(start)
        (e1, m1), (e2, m2) = chains
        s1, s2 = starts
        label = label[s1:s1 + len(m1), s2:s2 + len(m2)].copy()
        dist = dist[s1:s1 + len(m1), s2:s2 + len(m2)].copy()
        label[~m1.numpy()] = -1
        label[:, ~m2.numpy()] = -1

        if self.train:
            if random.random() < 0.5:  # chain order carries no meaning
                e1, m1, e2, m2 = e2, m2, e1, m1
                label, dist = label.T.copy(), dist.T.copy()
            if self.crop:
                i, j = self.crop_starts(label)
                e1, m1, label, dist = e1[i:i + self.crop], m1[i:i + self.crop], label[i:i + self.crop], dist[i:i + self.crop]
                e2, m2, label, dist = e2[j:j + self.crop], m2[j:j + self.crop], label[:, j:j + self.crop], dist[:, j:j + self.crop]
        return e1, m1, e2, m2, torch.from_numpy(np.ascontiguousarray(label)), torch.from_numpy(np.ascontiguousarray(dist))

    def crop_starts(self, label):
        """Window starts for both chains; around a random contact with probability 0.8."""
        L1, L2 = label.shape
        pos = np.argwhere(label == 1)
        if len(pos) and random.random() < 0.8:
            ci, cj = pos[random.randrange(len(pos))]
        else:
            ci, cj = random.randrange(L1), random.randrange(L2)

        def start(L, c):
            if L <= self.crop:
                return 0
            return random.randint(max(0, c - self.crop + 1), min(c, L - self.crop))
        return start(L1, ci), start(L2, cj)


class OnePerCluster(torch.utils.data.Sampler):
    """Each epoch: one random pair per cluster (PPLM clusters hold many near-duplicate pairs)."""

    def __init__(self, data):
        self.groups = {}
        for k, (_, _, cluster) in enumerate(data.index):
            self.groups.setdefault(cluster, []).append(k)

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        picks = [random.choice(g) for g in self.groups.values()]
        random.shuffle(picks)
        return iter(picks)


def collate(items):
    B = len(items)
    L1 = max(len(it[1]) for it in items)
    L2 = max(len(it[3]) for it in items)
    K, D = items[0][0].shape[1:]
    e1 = torch.zeros(B, L1, K, D, dtype=items[0][0].dtype)
    e2 = torch.zeros(B, L2, K, D, dtype=items[0][0].dtype)
    m1 = torch.zeros(B, L1, dtype=torch.bool)
    m2 = torch.zeros(B, L2, dtype=torch.bool)
    label = torch.full((B, L1, L2), -1, dtype=torch.int8)
    dist = torch.full((B, L1, L2), float("inf"))
    for b, (a1, n1, a2, n2, lab, d) in enumerate(items):
        e1[b, :len(n1)], m1[b, :len(n1)] = a1, n1
        e2[b, :len(n2)], m2[b, :len(n2)] = a2, n2
        label[b, :lab.shape[0], :lab.shape[1]] = lab
        dist[b, :d.shape[0], :d.shape[1]] = d
    return e1, m1, e2, m2, label, dist


# ---------------------------------------------------------------- model
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels at each (i, j): no statistics shared across positions or padding."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, h, dilation, dropout):
        super().__init__()
        self.norm = LayerNorm2d(h)
        self.conv = nn.Conv2d(h, h, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, z, mask):
        return (z + self.drop(self.conv(F.relu(self.norm(z))))) * mask


class ContactHead(nn.Module):
    def __init__(self, n_layers, dim=1280, hidden=128, blocks=4, pair="add", dropout=0.1):
        super().__init__()
        self.pair = pair
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.proj1 = nn.Linear(dim, hidden)
        self.proj2 = nn.Linear(dim, hidden, bias=pair == "multiply")  # when adding, proj1's bias covers both
        self.pair_norm = nn.LayerNorm(hidden)
        self.blocks = nn.ModuleList(ResBlock(hidden, 2 ** (b % 4), dropout) for b in range(blocks))
        self.out_norm = LayerNorm2d(hidden)
        self.out = nn.Conv2d(hidden, 2, 3, padding=1)

    def mix(self, e):
        """[B, L, K, D] -> [B, L, D]: normalise each layer (their scales differ), then mix."""
        e = F.layer_norm(e.float(), e.shape[-1:])
        return torch.einsum("blkd,k->bld", e, self.layer_logits.softmax(0))

    def forward(self, e1, m1, e2, m2):
        a, b = self.proj1(self.mix(e1)), self.proj2(self.mix(e2))   # [B, L1, h], [B, L2, h]
        if self.pair == "add":
            z = a[:, :, None] + b[:, None]                            # [B, L1, L2, h]
        else:
            z = a[:, :, None] * b[:, None]
        mask = (m1[:, :, None] & m2[:, None]).unsqueeze(1).to(z.dtype)  # [B, 1, L1, L2]
        z = F.relu(self.pair_norm(z)).permute(0, 3, 1, 2) * mask
        for block in self.blocks:
            z = block(z, mask)
        out = self.out(F.relu(self.out_norm(z)))
        return out[:, 0], out[:, 1]  # contact logits, distance / cutoff


# ---------------------------------------------------------------- loss / metrics
def compute_loss(logits, dist_pred, label, dist, cutoff, pos_weight, dist_weight):
    valid = label >= 0
    target = (label == 1).float()
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device))
    n = valid.sum().clamp(min=1)
    loss_contact = (bce * valid).sum() / n
    loss_dist = torch.zeros((), device=logits.device)
    if dist_weight:
        d = torch.nan_to_num(dist.clamp(max=cutoff), posinf=cutoff) / cutoff
        loss_dist = (((dist_pred.float() - d) ** 2) * valid).sum() / n
    return loss_contact + dist_weight * loss_dist, loss_contact, loss_dist


class Metrics:
    """Average precision from a probability histogram (no need to keep every cell) + top-k precision."""

    def __init__(self, bins=10000):
        self.bins = bins
        self.pos = np.zeros(bins, dtype=np.int64)
        self.neg = np.zeros(bins, dtype=np.int64)
        self.topk = {"top10": [], "topL/10": [], "topL/5": []}

    def add(self, probs, label):
        for p, lab in zip(probs, label):
            valid = lab >= 0
            if not valid.any():
                continue
            pv, lv = p[valid], lab[valid] == 1
            b = np.minimum((pv * self.bins).astype(int), self.bins - 1)
            self.pos += np.bincount(b[lv], minlength=self.bins)
            self.neg += np.bincount(b[~lv], minlength=self.bins)
            L = min(lab.shape)
            order = np.argsort(-pv)
            for name, k in (("top10", 10), ("topL/10", max(1, L // 10)), ("topL/5", max(1, L // 5))):
                self.topk[name].append(lv[order[:k]].mean())

    def result(self):
        tp, fp = np.cumsum(self.pos[::-1]), np.cumsum(self.neg[::-1])
        P = tp[-1]
        precision = tp / np.maximum(tp + fp, 1)
        recall_step = np.diff(np.r_[0, tp]) / max(P, 1)
        out = {"ap": float((precision * recall_step).sum()), "pos_rate": float(P / max(tp[-1] + fp[-1], 1))}
        out.update({k: float(np.mean(v)) if v else float("nan") for k, v in self.topk.items()})
        return out


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    metrics, total, n = Metrics(), 0.0, 0
    for e1, m1, e2, m2, label, dist in loader:
        e1, m1, e2, m2 = e1.to(device), m1.to(device), e2.to(device), m2.to(device)
        label, dist = label.to(device), dist.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, dist_pred = model(e1, m1, e2, m2)
        loss, _, _ = compute_loss(logits, dist_pred, label, dist, args.cutoff, args.pos_weight, args.dist_weight)
        total, n = total + loss.item(), n + 1
        metrics.add(torch.sigmoid(logits.float()).cpu().numpy(), label.cpu().numpy())
    return {"loss": total / max(n, 1), **metrics.result()}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-emb", required=True, help="shard folder from extract_residue_embeddings.py")
    ap.add_argument("--train-contacts", required=True, help="label_contacts.py JSONL for the same pairs")
    ap.add_argument("--val-emb", required=True)
    ap.add_argument("--val-contacts", required=True)
    ap.add_argument("--out", required=True, help="folder for best.pt, last.pt and log.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", help="saved layers to use (default: all saved)")
    ap.add_argument("--pair", choices=["add", "multiply"], default="add",
                    help="combine the two residues' projections by adding (= linear layer on the "
                         "concatenated pair) or multiplying (TEIM)")
    ap.add_argument("--hidden", type=int, default=128, help="pair-map channels; memory scales with it")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--crop", type=int, default=256, help="training crop per chain")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--pos-weight", type=float, default=1.0, help="BCE weight on contacts (they are ~1%% of cells)")
    ap.add_argument("--dist-weight", type=float, default=0.0, help="weight of the distance loss (0 = contacts only)")
    ap.add_argument("--cutoff", type=float, default=8.0, help="Å; contact definition, must match label_contacts.py")
    ap.add_argument("--all-pairs", action="store_true", help="use every pair each epoch, not one per cluster")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    train = PairData(args.train_emb, args.train_contacts, args.layers, args.crop, train=True)
    val = PairData(args.val_emb, args.val_contacts, train.layers)
    for name, data in (("train", train), ("val", val)):
        stored = {r["cutoff"] for r in data.contacts.values()}
        if any(c < args.cutoff for c in stored):
            raise SystemExit(f"{name} contacts were stored up to {min(stored)} Å < --cutoff {args.cutoff}")
    sampler = None if args.all_pairs else OnePerCluster(train)
    train_loader = DataLoader(train, args.batch_size, shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate, num_workers=args.workers, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, 1, collate_fn=collate, num_workers=args.workers)
    print(f"train {len(train)} pairs ({len(train_loader)} batches/epoch), val {len(val)} pairs, "
          f"layers {train.layers}")

    model = ContactHead(len(train.layers), train.dim, args.hidden, args.blocks, args.pair, args.dropout).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup = min(500, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / max(warmup, 1)) * 0.5 * (1 + math.cos(math.pi * min(s / max(total_steps, 1), 1))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best = -1
    for epoch in range(args.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for e1, m1, e2, m2, label, dist in train_loader:
            e1, m1, e2, m2 = e1.to(device), m1.to(device), e2.to(device), m2.to(device)
            label, dist = label.to(device), dist.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, dist_pred = model(e1, m1, e2, m2)
            loss, _, _ = compute_loss(logits, dist_pred, label, dist, args.cutoff, args.pos_weight, args.dist_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run, n = run + loss.item(), n + 1

        res = {"epoch": epoch, "train_loss": run / max(n, 1), **{f"val_{k}": v for k, v in evaluate(model, val_loader, device, args).items()},
               "layer_weights": model.layer_logits.softmax(0).tolist(), "minutes": (time.time() - t0) / 60}
        print(json.dumps(res))
        with open(os.path.join(args.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")
        ckpt = {"model": model.state_dict(), "args": vars(args), "layers": train.layers, "dim": train.dim, "epoch": epoch, "val": res}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if res["val_ap"] > best:
            best = res["val_ap"]
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
    print(f"best val AP {best:.4f}")


if __name__ == "__main__":
    main()
