#!/usr/bin/env python3
"""
Make per-residue interface labels for protein dimers from PDB structures.

Input (either format):
  * PPLM pair lists (trainset.PDB.txt / validset.PDB.txt): one cluster per line,
    pairs named  <pdb_id>-assembly<N>_<chain1>_<chain2>,  e.g.  5hrz-assembly1_B_C
  * a CSV with columns  pdb_id,chain1,chain2  (asymmetric-unit chain IDs)

Output: JSONL, one line per pair:
    {"id": "5hrz-assembly1_B_C", "cluster": 17,
     "seq1": "...", "seq2": "...", "labels1": [...], "labels2": [...]}

Labels, one per residue of seq1 / seq2:
     1 = interface residue (any heavy atom within CUTOFF Å of a partner heavy atom)
     0 = resolved in the structure, not at the interface
    -1 = not resolved / not alignable (ignore in the loss and in metrics)

With --seq-dir pointing at PPLM's PDB_seqs folder, seq1/seq2 are the sequences from
<pair-chain>.fa and labels are aligned onto them. Without it, the full deposited
(entity / SEQRES) sequence is used.

Setup (separate, modern env; not the Python 3.7 MINT env):
    pip install gemmi scipy numpy

Usage (from interaction_head/, after ./load_data.sh):
    python label_interfaces.py data/PPLM_Dataset/trainset.PDB.txt data/trainset.json \\
        --seq-dir data/PPLM_Dataset/PDB_seqs
    python label_interfaces.py dimers.csv data/labels.json --cutoff 8.0
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

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

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


def interface_labels(seq, atom_pos, coords_self, coords_partner, cutoff):
    labels = np.full(len(seq), -1, dtype=int)
    labels[np.unique(atom_pos)] = 0  # resolved residues
    dist, _ = cKDTree(coords_partner).query(coords_self, distance_upper_bound=cutoff)
    labels[np.unique(atom_pos[dist <= cutoff])] = 1
    return labels


# ---------------------------------------------------------------- sequence mapping
def read_fasta(path):
    with open(path) as f:
        return "".join(l.strip() for l in f if not l.startswith(">")).upper()


def transfer_labels(struct_seq, labels, target_seq):
    """Map labels from the structure's sequence onto target_seq via alignment."""
    if struct_seq == target_seq:
        return labels
    aln = gemmi.align_string_sequences(list(struct_seq), list(target_seq), [])
    a, b = aln.add_gaps(struct_seq, 1), aln.add_gaps(target_seq, 2)
    out = np.full(len(target_seq), -1, dtype=int)
    i = j = 0
    for ca, cb in zip(a, b):
        if ca != "-" and cb != "-":
            if ca == cb or "X" in (ca, cb):  # only trust identical (or unknown) positions
                out[j] = labels[i]
        if ca != "-":
            i += 1
        if cb != "-":
            j += 1
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PPLM pair list (.txt) or CSV with pdb_id,chain1,chain2")
    ap.add_argument("output_jsonl")
    ap.add_argument("--cutoff", type=float, default=8.0, help="Å; 8.0 matches PPLM-Contact")
    ap.add_argument("--seq-dir", help="PPLM PDB_seqs folder; labels are aligned onto its .fa sequences")
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
                lab1 = interface_labels(seq1, pos1, xyz1, xyz2, args.cutoff)
                lab2 = interface_labels(seq2, pos2, xyz2, xyz1, args.cutoff)

                if args.seq_dir and p["assembly"]:
                    prefix = f"{p['pdb_id']}-assembly{p['assembly']}_"
                    fa1 = read_fasta(os.path.join(args.seq_dir, prefix + p["chain1"] + ".fa"))
                    fa2 = read_fasta(os.path.join(args.seq_dir, prefix + p["chain2"] + ".fa"))
                    lab1, seq1 = transfer_labels(seq1, lab1, fa1), fa1
                    lab2, seq2 = transfer_labels(seq2, lab2, fa2), fa2

                if lab1.max() < 1 or lab2.max() < 1:
                    raise ValueError("no interface found at this cutoff")
                fout.write(json.dumps({
                    "id": p["name"], "cluster": p["cluster"],
                    "seq1": seq1, "seq2": seq2,
                    "labels1": lab1.tolist(), "labels2": lab2.tolist(),
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
