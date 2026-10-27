#!/usr/bin/env python3
"""
Make sparse inter-chain residue distance maps (L1 x L2) for protein dimers from PDB structures.

Input (either format):
  * PPLM pair lists (trainset.PDB.txt / validset.PDB.txt): one cluster per line,
    pairs named  <pdb_id>-assembly<N>_<chain1>_<chain2>,  e.g.  5hrz-assembly1_B_C
  * a CSV with columns  pdb_id,chain1,chain2  (asymmetric-unit chain IDs)

Output: JSONL, one line per pair:
    {"id": "5hrz-assembly1_B_C", "cluster": 17, "cutoff": 8.0,
     "chains": ["A", "A-2"], "chain_match": "order",
     "seq1": "...", "seq2": "...", "resolved1": [...], "resolved2": [...],
     "contacts": [[i, j, d], ...]}

contacts lists every residue pair (0-based i into seq1, j into seq2) whose minimum heavy-atom
distance d (Å) is <= cutoff; pairs farther apart are left out. resolved1 / resolved2 are 1/0
per residue of seq1 / seq2, so "far apart" (both resolved, not listed) can be told apart from
"unknown" (either unresolved / not alignable; ignore in the loss and in metrics).

Rebuild dense (len(seq1), len(seq2)) arrays with
    distance_map(record)          d where listed, inf where farther than cutoff, NaN where unknown
    contact_map(record, cutoff)   1 = within cutoff, 0 = farther, -1 = unknown
(contact_map's cutoff defaults to the stored one and can only be lowered.)

PPLM chain IDs are not the chain names in the RCSB assembly file: PPLM numbered the protein
chains of each assembly A..Z, 0..9 in file order (so 6j5g's A, H, L, A-2, H-2, L-2 are PPLM's
A..F), though not always. Pairs from a PPLM list are mapped that way; CSV chain IDs are used
as given.

With --seq-dir pointing at PPLM's PDB_seqs folder, seq1/seq2 are the sequences from
<pair-chain>.fa and the map is aligned onto them. If a mapped chain's resolved residues don't
match its .fa (modified and unknown residues such as MSE or UNK are not compared: PPLM's .fa
sequences leave them out, so they end up unknown in the map), the pair is instead taken from
the chains that do: of those pairs, the one with the most contacts (for homo-oligomers this
may be a different interface than PPLM's). A chain that is mostly UNK / modified can't be
checked and never matches.
Without --seq-dir, the full deposited (entity / SEQRES) sequence is used and the mapping is
unchecked. Each record's "chains" gives the file chains used and "chain_match" how they were
picked ("order", "sequence" or, for CSV input, "given").

Setup (separate, modern env; not the Python 3.7 MINT env):
    pip install gemmi scipy numpy

--per-cluster N labels at most N random pairs from each PPLM cluster (pairs in a cluster are
near-duplicates; --seed picks which). A pair that fails is replaced by another from its cluster.

Runs are resumable: re-running with the same output file skips pairs already labelled or
already listed in <output>.failed.txt (--retry-failed tries those again), so an interrupted run
can just be started again. The settings are saved to <output>.meta.json, and resuming with
different ones is refused.

Usage (from interaction_head/, after ./load_data.sh):
    python contact/label_contacts.py data/PPLM_Dataset/trainset.PDB.txt data/trainset_contacts.json \\
        --seq-dir data/PPLM_Dataset/PDB_seqs --per-cluster 1
    python contact/label_contacts.py dimers.csv data/contacts.json --cutoff 20.0
"""
import argparse
import csv
import json
import os
import random
import re
import urllib.request
from collections import Counter

import gemmi
import numpy as np
from scipy.spatial import cKDTree

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # interaction_head/data

PAIR_RE =re.compile(r"([0-9A-Za-z]{4})-assembly(\d+)_([^_\s,;]+)_([^_\s,;]+)")
PPLM_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # PPLM's k-th protein chain of an assembly
MIN_SEQ_MATCH = 0.9  # fraction of a chain's resolved standard residues that must align identically to its .fa


# ---------------------------------------------------------------- input parsing
def read_pairs(path, per_cluster=None, seed=0):
    """Yield dicts: name, pdb_id, assembly (None = asymmetric unit), chain1, chain2, cluster."""
    if path.lower().endswith(".csv"):
        with open(path) as f:
            for i, row in enumerate(csv.DictReader(f)):
                pdb_id, c1, c2 = row["pdb_id"].strip(), row["chain1"].strip(), row["chain2"].strip()
                yield dict(name=f"{pdb_id}_{c1}_{c2}", pdb_id=pdb_id, assembly=None,
                           chain1=c1, chain2=c2, cluster=i)
        return
    rng = random.Random(seed)
    with open(path) as f:
        for cluster, line in enumerate(l for l in f if l.strip()):
            pairs = list(PAIR_RE.finditer(line))
            if per_cluster:
                rng.shuffle(pairs)  # main() takes the first per_cluster that succeed
            for m in pairs:
                yield dict(name=m.group(0), pdb_id=m.group(1), assembly=m.group(2),
                           chain1=m.group(3), chain2=m.group(4), cluster=cluster)


# ---------------------------------------------------------------- structures
def fetch_cif(pdb_id, assembly, cache_dir):
    """Download the mmCIF (biological assembly if given) from RCSB once and cache it."""
    os.makedirs(cache_dir, exist_ok=True)
    stem = f"{pdb_id.lower()}-assembly{assembly}" if assembly else pdb_id.lower()
    path = os.path.join(cache_dir, f"{stem}.cif.gz")
    if not os.path.exists(path):
        url_stem = f"{pdb_id.lower()}-assembly{assembly}" if assembly else pdb_id.upper()
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{url_stem}.cif.gz", path + ".part")
        os.replace(path + ".part", path)  # an interrupted download never looks complete
    return path


def load_structure(path):
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    return st


def pplm_chain_names(st):
    """PPLM chain ID -> chain name in the assembly file (protein chains numbered in file order)."""
    protein = [ch.name for ch in st[0] if ch.get_polymer().check_polymer_type()
               in (gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD)]
    return dict(zip(PPLM_CHAIN_IDS, protein))


def one_letter(resname):
    """3-letter residue name -> 1-letter code; modified residues map to their parent (MSE -> M)."""
    info = gemmi.find_tabulated_residue(resname)
    code = info.one_letter_code.upper() if info else "X"
    return code if code.isalpha() else "X"


def is_standard(resname):
    """False for modified (MSE, TPO, SEP, ...) and unknown (UNK) residues, which PPLM's .fa
    sequences leave out."""
    info = gemmi.find_tabulated_residue(resname)
    if info is None:
        return True  # name not in gemmi's table: can't tell, so count it
    return info.is_standard() and resname != "UNK"


def chain_data(st, chain_id):
    """Return (full entity sequence, per-position standard-residue mask, heavy-atom coords,
    sequence index of each atom)."""
    chain = st[0].find_chain(chain_id)
    if chain is None:
        raise KeyError(f"chain {chain_id} not found")
    polymer = chain.get_polymer()
    entity = st.get_entity_of(polymer)
    if entity is None or not entity.full_sequence:
        raise ValueError(f"no entity sequence for chain {chain_id}")

    mons = [gemmi.Entity.first_mon(m) for m in entity.full_sequence]
    seq = "".join(one_letter(m) for m in mons)
    standard = np.array([is_standard(m) for m in mons])

    coords, atom_pos = [], []
    for res in polymer:
        if res.label_seq is None:
            continue
        idx = res.label_seq - 1  # label_seq is 1-based into the entity sequence
        for atom in res:
            coords.append(atom.pos.tolist())
            atom_pos.append(idx)
    if not coords:
        raise ValueError(f"no atoms in chain {chain_id}")
    return seq, standard, np.asarray(coords), np.asarray(atom_pos)


def resolved_mask(seq, atom_pos):
    resolved = np.zeros(len(seq), dtype=int)
    resolved[np.unique(atom_pos)] = 1
    return resolved


def residue_distances(pos1, coords1, pos2, coords2, cutoff):
    """Residue pairs within cutoff: (ij, d) with ij of shape (N, 2) and d the min heavy-atom distance."""
    pairs = cKDTree(coords1).sparse_distance_matrix(cKDTree(coords2), cutoff, output_type="ndarray")
    if not len(pairs):
        return np.zeros((0, 2), dtype=int), np.zeros(0)
    r1, r2, d = pos1[pairs["i"]], pos2[pairs["j"]], pairs["v"]
    order = np.lexsort((r2, r1))  # group atom pairs by residue pair, then take each group's min
    r1, r2, d = r1[order], r2[order], d[order]
    start = np.flatnonzero(np.r_[True, (np.diff(r1) != 0) | (np.diff(r2) != 0)])
    return np.stack([r1[start], r2[start]], axis=1), np.minimum.reduceat(d, start)


def distance_map(record):
    """Dense (L1, L2) map from an output record: d if listed, inf if beyond cutoff, NaN if unknown."""
    r1 = np.asarray(record["resolved1"], dtype=bool)
    r2 = np.asarray(record["resolved2"], dtype=bool)
    dist = np.full((len(r1), len(r2)), np.nan, dtype=np.float32)
    dist[np.ix_(r1, r2)] = np.inf
    if record["contacts"]:
        c = np.asarray(record["contacts"])
        dist[c[:, 0].astype(int), c[:, 1].astype(int)] = c[:, 2]
    return dist


def contact_map(record, cutoff=None):
    """Dense (L1, L2) labels: 1 = within cutoff (default: the stored one), 0 = farther, -1 = unknown."""
    cutoff = record["cutoff"] if cutoff is None else cutoff
    if cutoff > record["cutoff"]:
        raise ValueError(f"record only has distances up to {record['cutoff']} Å")
    dist = distance_map(record)
    cmap = np.where(dist <= cutoff, 1, 0).astype(np.int8)
    cmap[np.isnan(dist)] = -1
    return cmap


# ---------------------------------------------------------------- sequence mapping
def read_fasta(path):
    with open(path) as f:
        return "".join(l.strip() for l in f if not l.startswith(">")).upper()


def align_index(struct_seq, target_seq):
    """For each struct_seq position, its index in target_seq via alignment (-1 = unmapped)."""
    if struct_seq == target_seq:
        return np.arange(len(struct_seq))
    aln = gemmi.align_string_sequences(list(struct_seq), list(target_seq), [])
    a, b = aln.add_gaps(struct_seq, 1), aln.add_gaps(target_seq, 2)
    out = np.full(len(struct_seq), -1, dtype=int)
    i = j = 0
    for ca, cb in zip(a, b):
        if ca != "-" and cb != "-":
            if ca == cb or "X" in (ca, cb):  # only trust identical (or unknown) positions
                out[i] = j
        if ca != "-":
            i += 1
        if cb != "-":
            j += 1
    return out


def seq_match(checked, index):
    """Fraction of the checked positions (resolved, standard residues) that align identically
    to the target sequence; 0 if there are none."""
    checked = checked.astype(bool)
    return (index[checked] >= 0).mean() if checked.any() else 0.0


def transfer_resolved(resolved, index, target_len):
    out = np.zeros(target_len, dtype=int)
    ok = index >= 0
    out[index[ok]] = resolved[ok]
    return out


def transfer_contacts(ij, d, index1, index2):
    i, j = index1[ij[:, 0]], index2[ij[:, 1]]
    ok = (i >= 0) & (j >= 0)
    return np.stack([i[ok], j[ok]], axis=1), d[ok]


# ---------------------------------------------------------------- chain choice
class ChainMismatch(ValueError):
    pass


class Assembly:
    """A loaded structure plus per-chain data, computed once per chain."""
    def __init__(self, st, pplm_ids):
        self.st = st
        self.names = pplm_chain_names(st) if pplm_ids else None
        self._chains, self._matches = {}, {}

    def chain(self, name):
        """(seq, coords, atom_pos, resolved, checked) of a chain; checked = resolved standard
        residues, the ones expected in the .fa (it leaves modified residues out)."""
        if name not in self._chains:
            seq, standard, xyz, pos = chain_data(self.st, name)
            res = resolved_mask(seq, pos)
            checked = res.astype(bool) & standard
            if checked.sum() < 0.5 * res.sum():
                checked[:] = False  # mostly UNK / modified: too little to tell which chain it is
            self._chains[name] = seq, xyz, pos, res, checked
        return self._chains[name]

    def matching_chains(self, fa):
        """Protein chains, in file order, whose resolved standard residues align to fa."""
        if fa not in self._matches:
            out = []
            for name in self.names.values():
                try:
                    seq, _, _, _, checked = self.chain(name)
                except (KeyError, ValueError):
                    continue
                if seq_match(checked, align_index(seq, fa)) >= MIN_SEQ_MATCH:
                    out.append(name)
            self._matches[fa] = out
        return self._matches[fa]


def label_chains(asm, c1, c2, fa1, fa2, cutoff):
    """(seq1, seq2, res1, res2, ij, d) for file chains c1, c2, aligned onto fa1 / fa2 unless None."""
    seq1, xyz1, pos1, res1, chk1 = asm.chain(c1)
    seq2, xyz2, pos2, res2, chk2 = asm.chain(c2)
    if fa1 is not None:
        idx1, idx2 = align_index(seq1, fa1), align_index(seq2, fa2)
        for c, chk, idx in ((c1, chk1, idx1), (c2, chk2, idx2)):
            if (frac := seq_match(chk, idx)) < MIN_SEQ_MATCH:
                raise ChainMismatch(f"chain {c} doesn't match its .fa sequence "
                                    f"({frac:.0%} of resolved standard residues align)")
    ij, d = residue_distances(pos1, xyz1, pos2, xyz2, cutoff)
    if fa1 is not None:
        res1, res2 = transfer_resolved(res1, idx1, len(fa1)), transfer_resolved(res2, idx2, len(fa2))
        ij, d = transfer_contacts(ij, d, idx1, idx2)
        seq1, seq2 = fa1, fa2
    return seq1, seq2, res1, res2, ij, d


def near(asm, c1, c2, cutoff):
    """Cheap pre-check: do the two chains' bounding boxes come within cutoff?"""
    a, b = asm.chain(c1)[1], asm.chain(c2)[1]
    return np.all(a.min(0) - cutoff <= b.max(0)) and np.all(b.min(0) - cutoff <= a.max(0))


def label_by_sequence(asm, fa1, fa2, cutoff):
    """Fallback when PPLM's chain order can't be reproduced: of all chain pairs matching fa1 / fa2,
    the one with the most contacts (first in file order on ties)."""
    best = None
    for c1 in asm.matching_chains(fa1):
        for c2 in asm.matching_chains(fa2):
            if c1 == c2 or not near(asm, c1, c2, cutoff):
                continue
            out = label_chains(asm, c1, c2, fa1, fa2, cutoff)
            if best is None or len(out[4]) > len(best[1][4]):
                best = (c1, c2), out
    if best is None:
        raise ChainMismatch("no pair of chains matching the .fa sequences is in contact")
    return best


# ---------------------------------------------------------------- resuming
def resume(out_path, failed_path, meta, retry_failed):
    """(ids an earlier run into out_path already handled, labelled pairs per cluster); empty for a new run."""
    meta_path = out_path + ".meta.json"
    if not os.path.exists(out_path):
        if os.path.exists(failed_path):
            os.remove(failed_path)  # don't leave a stale report from an earlier run
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return set(), Counter()

    if not os.path.exists(meta_path):
        raise SystemExit(f"{out_path} exists but has no {os.path.basename(meta_path)} (made by an older "
                         "version of this script); delete it or write to a new file")
    with open(meta_path) as f:
        old = json.load(f)
    diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
    if diff:
        raise SystemExit(f"{out_path} was made with other settings (old, new): {diff}; "
                         "delete it or write to a new file")

    done, per_cluster, good = set(), Counter(), 0
    with open(out_path, "rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                break  # cut off by a crash
            rec = json.loads(line)
            done.add(rec["id"])
            per_cluster[rec["cluster"]] += 1
            good += len(line)
    os.truncate(out_path, good)
    n_failed = 0
    if os.path.exists(failed_path):
        if retry_failed:
            os.remove(failed_path)
        else:
            with open(failed_path) as f:
                failed = {l.split("\t")[0] for l in f if l.endswith("\n")}
            n_failed = len(failed)
            done |= failed
    print(f"resuming: {len(done) - n_failed} pairs already labelled, {n_failed} failed earlier"
          + (" (skipped; --retry-failed to try again)" if n_failed else ""))
    return done, per_cluster


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PPLM pair list (.txt) or CSV with pdb_id,chain1,chain2")
    ap.add_argument("output_jsonl")
    ap.add_argument("--cutoff", type=float, default=8.0,
                    help="Å; only residue pairs this close are stored (8.0 matches PPLM-Contact)")
    ap.add_argument("--seq-dir", help="PPLM PDB_seqs folder; maps are aligned onto its .fa sequences")
    ap.add_argument("--per-cluster", type=int, help="keep at most N random pairs per PPLM cluster (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="which pairs --per-cluster keeps")
    ap.add_argument("--retry-failed", action="store_true", help="when resuming, retry pairs that failed before")
    ap.add_argument("--cache", default=os.path.join(DATA_DIR, "cif_cache"))
    args = ap.parse_args()

    failed_path = args.output_jsonl + ".failed.txt"
    meta = {"input": os.path.abspath(args.input), "cutoff": args.cutoff,
            "seq_dir": args.seq_dir and os.path.abspath(args.seq_dir),
            "per_cluster": args.per_cluster, "seed": args.seed if args.per_cluster else None,
            "chain_ids": "pplm-order"}  # older outputs looked PPLM chain IDs up by name
    done, n_in_cluster = resume(args.output_jsonl, failed_path, meta, args.retry_failed)

    n_ok = n_failed = 0
    structures = {}
    with open(args.output_jsonl, "a") as fout, open(failed_path, "a") as ffail:
        for p in read_pairs(args.input, args.per_cluster, args.seed):
            if p["name"] in done or (args.per_cluster and n_in_cluster[p["cluster"]] >= args.per_cluster):
                continue
            try:
                key = (p["pdb_id"].lower(), p["assembly"])
                if key not in structures:  # pairs from one entry share the download
                    structures.clear()
                    st = load_structure(fetch_cif(p["pdb_id"], p["assembly"], args.cache))
                    structures[key] = Assembly(st, pplm_ids=bool(p["assembly"]))
                asm = structures[key]

                fa1 = fa2 = None
                if args.seq_dir and p["assembly"]:
                    prefix = f"{p['pdb_id']}-assembly{p['assembly']}_"
                    fa1 = read_fasta(os.path.join(args.seq_dir, prefix + p["chain1"] + ".fa"))
                    fa2 = read_fasta(os.path.join(args.seq_dir, prefix + p["chain2"] + ".fa"))

                c1, c2, how = p["chain1"], p["chain2"], "given"
                if asm.names is not None:
                    c1, c2, how = asm.names.get(c1), asm.names.get(c2), "order"
                try:
                    if c1 is None or c2 is None:
                        raise ChainMismatch(f"PPLM chain ID beyond the assembly's {len(asm.names)} protein chains")
                    seq1, seq2, res1, res2, ij, d = label_chains(asm, c1, c2, fa1, fa2, args.cutoff)
                except ChainMismatch as e:
                    if fa1 is None:
                        raise
                    try:
                        (c1, c2), (seq1, seq2, res1, res2, ij, d) = label_by_sequence(asm, fa1, fa2, args.cutoff)
                    except ChainMismatch as e2:
                        raise ChainMismatch(f"{e}; {e2}") from None
                    how = "sequence"

                if not len(ij):
                    raise ValueError("no contacts found at this cutoff")
                fout.write(json.dumps({
                    "id": p["name"], "cluster": p["cluster"], "cutoff": args.cutoff,
                    "chains": [c1, c2], "chain_match": how,
                    "seq1": seq1, "seq2": seq2,
                    "resolved1": res1.tolist(), "resolved2": res2.tolist(),
                    "contacts": [[int(i), int(j), round(float(x), 2)] for (i, j), x in zip(ij, d)],
                }) + "\n")
                fout.flush()  # so an interrupted run loses at most the pair in progress
                n_ok += 1
                n_in_cluster[p["cluster"]] += 1
            except Exception as e:  # keep going; report at the end
                ffail.write(f"{p['name']}\t{' '.join(str(e).split())}\n")
                ffail.flush()
                n_failed += 1

    print(f"labeled {n_ok} pairs, {n_failed} failed" + (f" (this run; {len(done)} done before)" if done else ""))
    if os.path.getsize(failed_path):
        print(f"see {failed_path}")
    else:
        os.remove(failed_path)


if __name__ == "__main__":
    main()
