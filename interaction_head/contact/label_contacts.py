#!/usr/bin/env python3
"""
Make sparse inter-chain residue distance maps (L1 x L2) for protein dimers from PDB structures.

Input (either format):
  * PPLM pair lists (trainset.PDB.txt / validset.PDB.txt): one cluster per line,
    pairs named  <pdb_id>-assembly<N>_<chain1>_<chain2>,  e.g.  5hrz-assembly1_B_C
  * a CSV with columns  pdb_id,chain1,chain2  (asymmetric-unit chain IDs)

Output: JSONL, one line per pair:
    {"id": "5hrz-assembly1_B_C", "cluster": 17, "cutoff": 8.0,
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

With --seq-dir pointing at PPLM's PDB_seqs folder, seq1/seq2 are the sequences from
<pair-chain>.fa and the map is aligned onto them. Without it, the full deposited
(entity / SEQRES) sequence is used.

Setup (separate, modern env; not the Python 3.7 MINT env):
    pip install gemmi scipy numpy

Usage (from interaction_head/, after ./load_data.sh):
    python contact/label_contacts.py data/PPLM_Dataset/trainset.PDB.txt data/trainset_contacts.json \\
        --seq-dir data/PPLM_Dataset/PDB_seqs
    python contact/label_contacts.py dimers.csv data/contacts.json --cutoff 20.0
"""
import argparse
import csv
import json
import os
import re
import urllib.request

import gemmi
import numpy as np
from scipy.spatial import cKDTree

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # interaction_head/data

PAIR_RE =re.compile(r"([0-9A-Za-z]{4})-assembly(\d+)_([^_\s,;]+)_([^_\s,;]+)")


# ---------------------------------------------------------------- input parsing
def read_pairs(path):
    """Yield dicts: name, pdb_id, assembly (None = asymmetric unit), chain1, chain2, cluster."""
    if path.lower().endswith(".csv"):
        with open(path) as f:
            for i, row in enumerate(csv.DictReader(f)):
                pdb_id, c1, c2 = row["pdb_id"].strip(), row["chain1"].strip(), row["chain2"].strip()
                yield dict(name=f"{pdb_id}_{c1}_{c2}", pdb_id=pdb_id, assembly=None,
                           chain1=c1, chain2=c2, cluster=i)
        return
    with open(path) as f:
        for cluster, line in enumerate(l for l in f if l.strip()):
            for m in PAIR_RE.finditer(line):
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
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{url_stem}.cif.gz", path)
    return path


def load_structure(path):
    st = gemmi.read_structure(path)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    return st


def one_letter(resname):
    """3-letter residue name -> 1-letter code; modified residues map to their parent (MSE -> M)."""
    info = gemmi.find_tabulated_residue(resname)
    code = info.one_letter_code.upper() if info else "X"
    return code if code.isalpha() else "X"


def chain_data(st, chain_id):
    """Return (full entity sequence, heavy-atom coords, sequence index of each atom)."""
    chain = st[0].find_chain(chain_id)
    if chain is None:
        raise KeyError(f"chain {chain_id} not found")
    polymer = chain.get_polymer()
    entity = st.get_entity_of(polymer)
    if entity is None or not entity.full_sequence:
        raise ValueError(f"no entity sequence for chain {chain_id}")

    seq = "".join(one_letter(gemmi.Entity.first_mon(m)) for m in entity.full_sequence)

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
    return seq, np.asarray(coords), np.asarray(atom_pos)


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


def transfer_resolved(resolved, index, target_len):
    out = np.zeros(target_len, dtype=int)
    ok = index >= 0
    out[index[ok]] = resolved[ok]
    return out


def transfer_contacts(ij, d, index1, index2):
    i, j = index1[ij[:, 0]], index2[ij[:, 1]]
    ok = (i >= 0) & (j >= 0)
    return np.stack([i[ok], j[ok]], axis=1), d[ok]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PPLM pair list (.txt) or CSV with pdb_id,chain1,chain2")
    ap.add_argument("output_jsonl")
    ap.add_argument("--cutoff", type=float, default=8.0,
                    help="Å; only residue pairs this close are stored (8.0 matches PPLM-Contact)")
    ap.add_argument("--seq-dir", help="PPLM PDB_seqs folder; maps are aligned onto its .fa sequences")
    ap.add_argument("--cache", default=os.path.join(DATA_DIR, "cif_cache"))
    args = ap.parse_args()

    failed_path = args.output_jsonl + ".failed.txt"
    if os.path.exists(failed_path):
        os.remove(failed_path)  # don't leave a stale report from an earlier run

    n_ok, failed, structures = 0, [], {}
    with open(args.output_jsonl, "w") as fout:
        for p in read_pairs(args.input):
            try:
                key = (p["pdb_id"].lower(), p["assembly"])
                if key not in structures:  # pairs from one entry share the download
                    structures.clear()
                    structures[key] = load_structure(fetch_cif(p["pdb_id"], p["assembly"], args.cache))
                st = structures[key]

                seq1, xyz1, pos1 = chain_data(st, p["chain1"])
                seq2, xyz2, pos2 = chain_data(st, p["chain2"])
                res1, res2 = resolved_mask(seq1, pos1), resolved_mask(seq2, pos2)
                ij, d = residue_distances(pos1, xyz1, pos2, xyz2, args.cutoff)

                if args.seq_dir and p["assembly"]:
                    prefix = f"{p['pdb_id']}-assembly{p['assembly']}_"
                    fa1 = read_fasta(os.path.join(args.seq_dir, prefix + p["chain1"] + ".fa"))
                    fa2 = read_fasta(os.path.join(args.seq_dir, prefix + p["chain2"] + ".fa"))
                    idx1, idx2 = align_index(seq1, fa1), align_index(seq2, fa2)
                    res1, res2 = transfer_resolved(res1, idx1, len(fa1)), transfer_resolved(res2, idx2, len(fa2))
                    ij, d = transfer_contacts(ij, d, idx1, idx2)
                    seq1, seq2 = fa1, fa2

                if not len(ij):
                    raise ValueError("no contacts found at this cutoff")
                fout.write(json.dumps({
                    "id": p["name"], "cluster": p["cluster"], "cutoff": args.cutoff,
                    "seq1": seq1, "seq2": seq2,
                    "resolved1": res1.tolist(), "resolved2": res2.tolist(),
                    "contacts": [[int(i), int(j), round(float(x), 2)] for (i, j), x in zip(ij, d)],
                }) + "\n")
                n_ok += 1
            except Exception as e:  # keep going; report at the end
                failed.append(f"{p['name']}\t{e}")

    print(f"labeled {n_ok} pairs, {len(failed)} failed")
    if failed:
        with open(failed_path, "w") as f:
            f.write("\n".join(failed) + "\n")
        print(f"see {failed_path}")


if __name__ == "__main__":
    main()
