#!/bin/bash
# Download the PPLM dataset (pair lists + PDB_seqs) into interaction_head/data/
set -e
cd "$(dirname "$0")/data"
wget https://zhanggroup.org/PPLM/bin/databases/PPLM_Dataset.tar.gz
tar -xzf PPLM_Dataset.tar.gz
rm PPLM_Dataset.tar.gz
