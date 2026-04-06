#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
try:
    import pandas as pd
    pandas_imported = True
except ImportError as e:
    pandas_imported = False
import numpy as np
import os, sys
import platform
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import Align
from typing import List, Sequence, Union
opsys = platform.system()
from project_config.variables import mapping, aaList_with_X, aaList


def normalize_sequence(sequence: str) -> str:
    """
    Upper-case a protein sequence and replace unsupported residues with `X`.
    """
    cleaned = "".join(str(sequence or "").upper().split())
    return "".join(res if res in set(aaList_with_X) else "X" for res in cleaned)


def normalize_sequences(sequences: Union[str, Sequence[str]]) -> List[str]:
    """
    Normalize one or more protein sequences into a clean non-empty list.
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    normalized = [normalize_sequence(seq) for seq in sequences]
    return [seq for seq in normalized if seq]


def get_mutstr(mutation):
    if isinstance(mutation, list):
        mutstr = '+'.join(mutation)
    else:
        mutstr = mutation
        mutation = [mutation]
    return mutstr, mutation


def get_mapping(mapping, res):
    # Map residue initial to code
    result = mapping[res]
    return result


def split_mutation(mutation, aa_letter_representation=False):
    # Convert point mutation to wildtype residue, muted residue and mutation position
    WT_res = mutation[0]
    MT_res = mutation[-1]
    if not aa_letter_representation:
        WT_res = get_mapping(mapping, WT_res)
        MUT_res = get_mapping(mapping, MT_res)
    pos = int(mutation[1:-1])
    return WT_res, pos, MT_res


def split_wildtype(mutation):
    # Convert point mutation to wildtype residue, muted residue and mutation position
    WT_res = mutation[0]
    MUT_pos = int(mutation[1:len(mutation)-1])
    MT_res = mutation[-1]
    return WT_res, MUT_pos, MT_res


def get_mutations(wildtype_list):
    # get amino acid list to perform mutations to
    aaList = ['A', 'H', 'Y', 'R', 'T', 'K', 'M', 'D', 'N', 'C', 'Q', 'E', 'G', 'I', 'L', 'F', 'P', 'S', 'W', 'V']
    # get all mutations to run
    mutations = []
    for wt in wildtype_list:
        wtAA = wt[0]
        for aa in aaList:
            if aa != wtAA:
                mt = wt + aa
                mutations.append(mt)
    print('mutant:', mutations)
    return mutations


def get_mutated_sequence(seq_base, mutations, seq_name=None, write_to_fasta=None):
    mutations_list = []
    seq_name_list = []
    sequence_list = []
    fasta_list = []

    # get mutated sequences
    for mut in mutations:
        if mut not in [None, 'WT']:
            wildtype_aa, position, mutant_aa = split_wildtype(mut)

            if seq_base[position-1] == wildtype_aa:
                if seq_name is None:
                    seq_name_wmut = mut
                else:
                    seq_name_wmut = seq_name + '_' + mut
            else:
                seq_name_wmut = seq_name + '_' + mut
            list_seq_base = list(seq_base)
            list_seq_base[position-1] = mutant_aa
            seq_mutate = "".join(list_seq_base)

        else:
            mut = 'WT'
            seq_mutate = seq_base
            seq_name_wmut = seq_name + '_' + mut

        mutations_list.append(mut)
        sequence_list.append(seq_mutate)
        seq_name_list.append(seq_name_wmut)

        # write to fasta
        if write_to_fasta is not None:
            fasta_file = write_sequence_to_fasta(seq_mutate, seq_name_wmut, seq_name_wmut, write_to_fasta)
            fasta_list.append(fasta_file)

    return mutations_list, seq_name_list, sequence_list, fasta_list


def fetch_sequences_from_fasta(sequence_fpath):

    import os
    sequence_names = []
    sequence_list = []
    sequence_descriptions = []
    for j, record in enumerate(SeqIO.parse(sequence_fpath, "fasta")):
        sequence_names.append(record.id)
        sequence_list.append(str(record.seq))
        sequence_descriptions.append(record.description)
    return sequence_list, sequence_names, sequence_descriptions


def write_sequence_to_fasta(sequences, seq_names, filename, fasta_dir):
    fasta_file = fasta_dir + filename
    if isinstance(sequences, str) and isinstance(seq_names, str):
        sequences = [sequences]
        seq_names = [seq_names]
    with open(fasta_file, 'w') as f:
        for i, (sequence, seq_name) in enumerate(zip(sequences, seq_names)):
            if sequence is not None:
                f.write('>' + seq_name + '\n')
                if i==len(sequences)-1:
                    f.write(sequence)
                else:
                    f.write(sequence+'\n')
    print('Saved fasta file to ' + fasta_file)
    return fasta_file


def split_fasta(max_res_per_fasta, fasta_fname, fasta_dir):
    seqs, seq_names, _ = fetch_sequences_from_fasta(fasta_dir+fasta_fname)
    num_seq = len(seqs)
    seq_len = len(seqs[0])
    print('# of seqs:', num_seq, '; seq len:', seq_len)
    num_seq_per_fasta = int(np.floor(max_res_per_fasta / seq_len))
    num_fasta_to_split = int(np.floor(num_seq / num_seq_per_fasta))
    print('# fasta files to split too:', num_fasta_to_split, '; # of seqs per fasta:', num_seq_per_fasta)

    for i in range(num_fasta_to_split):
        seqs_i = seqs[i * num_seq_per_fasta:min((i + 1) * num_seq_per_fasta, num_seq)]
        seq_names_i = seq_names[i * num_seq_per_fasta:min((i + 1) * num_seq_per_fasta, num_seq)]
        fasta_out = write_sequence_to_fasta(seqs_i, seq_names_i, fasta_fname.replace('.fasta','') + '_' + str(i), fasta_dir)


def deduplicate_fasta_sequences(fasta_fname_in, sort_by_seq_name=False, fasta_fname_out=None, fasta_dir='./'):
    seqs, seq_names, _ = fetch_sequences_from_fasta(fasta_dir+fasta_fname_in)
    seqs_deduped = []
    seq_names_deduped = []
    for seq, seq_name in zip(seqs, seq_names):
        if seq_name not in seq_names_deduped:
            seq_names_deduped.append(seq_name)
            seqs_deduped.append(seq)
    print('# of seqs [BEFORE]', len(seq_names), '[AFTER]', len(seq_names_deduped))

    # sort alphabetically by seq name
    if sort_by_seq_name:
        sort_idx = np.argsort(np.array(seq_names_deduped))
        seq_names_deduped = [seq_names_deduped[idx] for idx in sort_idx]
        seqs_deduped = [seqs_deduped[idx] for idx in sort_idx]

    # save resulting fasta
    if fasta_fname_out is None:
        fasta_fname_out = fasta_fname_in
    fasta_fname_out = fasta_fname_out.replace('.fasta','')
    fasta_out = write_sequence_to_fasta(seqs_deduped, seq_names_deduped, fasta_fname_out,  fasta_dir)
    return fasta_out, seqs_deduped, seq_names_deduped


def run_pairwise_alignment(seq1, seq2,
                           mode='global', match_score=2, mismatch_score=-1,
                           open_gap_score=-0.5, extend_gap_score=-0.1,
                           target_end_gap_score=0.0, query_end_gap_score=0.0, print_alignments=False):
    # initialize aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.match_score = match_score
    aligner.mismatch_score = mismatch_score
    aligner.open_gap_score = open_gap_score
    aligner.extend_gap_score = extend_gap_score
    aligner.target_end_gap_score = target_end_gap_score
    aligner.query_end_gap_score = query_end_gap_score
    # perform alignment
    alignments = aligner.align(seq1, seq2)
    for i, alignment in enumerate(alignments):
        if print_alignments:
            print('Alignment', i)
            print(alignment.aligned)
            print("Score = %.1f:" % alignment.score)
            print(alignment)
    return alignments


def get_mutagenesis_sequences(mutations_list, seq_backbone, split_on='_'):
    seq_mutants_list = []
    for mutations in mutations_list:
        seq_mutant = seq_backbone
        muts = mutations.split(split_on)
        for mut in muts:
            mt_aa = mut[-1]
            res = int(mut[1:-1])
            res_zeroidx = res-1
            seq_mutant = seq_mutant[:res_zeroidx] + mt_aa + seq_mutant[res_zeroidx+1:]
        seq_mutants_list.append(seq_mutant)
    return seq_mutants_list


def get_ref_seq_idxs_aa_from_msa(msa_path, ref_seq_name_list, zero_indexed=False):
    # get ref_seq_name_list found in MSA
    ref_seq_name_list_inmsa = []
    ref_seq_idxs_list_inmsa = []
    ref_seq_list_inmsa = []
    # check that all ref seqs are in the MSA
    msa_seqs, msa_names, _ = fetch_sequences_from_fasta(msa_path)
    for ref_seq_name in ref_seq_name_list:
        ref_seq_inmsa = ref_seq_name in msa_names
        print(ref_seq_name + ' is in MSA: ' + str(ref_seq_inmsa))
        if ref_seq_name in msa_names:
            ref_seq_name_list_inmsa.append(ref_seq_name)
            msa_idx = msa_names.index(ref_seq_name)
            msa_seq = msa_seqs[msa_idx]
            seq_filt = ''
            idx_filt = []
            for i, letter in enumerate(list(msa_seq)):
                if letter != '-':
                    seq_filt += letter
                    if zero_indexed:
                        idx_filt.append(i)
                    else:
                        idx_filt.append(i+1)
            ref_seq_idxs_list_inmsa.append(idx_filt)
            ref_seq_list_inmsa.append(seq_filt)
    return ref_seq_name_list_inmsa, ref_seq_list_inmsa, ref_seq_idxs_list_inmsa


def add_sequences_to_dataframe(csv_fpath, enzname_to_seq_dict, df=None):
    # get dataframe
    if df is None:
        df = pd.read_csv(csv_fpath, index_col=0)
    # get enzyme names
    enz_name_list = df['enz_name'].tolist()
    # match to sequence
    sequence_list = []
    for enz_name in enz_name_list:
        sequence_list.append(enzname_to_seq_dict[enz_name])
    # update df
    df['sequence'] =  sequence_list
    # save df
    df.to_csv(csv_fpath)
    return df

def parse_genbank_to_df(genbank_text):
    """
    Parses a GenBank text entry and converts it into a Pandas DataFrame.
    Extracts functional annotations, taxonomy, sequence metadata, and references.
    Parameters:
    - genbank_text: String containing the GenBank entry.
    Returns:
    - df: Pandas DataFrame containing parsed data.
    """
    from Bio import SeqIO
    from io import StringIO

    # Read the GenBank entry as a file-like object
    genbank_io = StringIO(genbank_text)

    # Parse the record using Biopython
    record = SeqIO.read(genbank_io, "genbank")

    # Extract relevant fields
    parsed_data = {
        "Locus": record.name,
        "Length (aa)": len(record.seq),
        "Type": record.annotations.get("molecule_type", "Unknown"),
        "Date": record.annotations.get("date", "Unknown"),
        "Accession": record.id,
        "Version": record.annotations.get("sequence_version", "Unknown"),
        "Definition": record.description,
        "Keywords": record.annotations.get("keywords", "None"),
        "Source": record.annotations.get("source", "Unknown"),
        "Organism": record.annotations.get("organism", "Unknown"),
        "Taxonomy": " > ".join(record.annotations.get("taxonomy", ["Unknown"])),
        "Enzyme Function": record.annotations.get("comment", "Not Specified"),
        "Protein Features": ", ".join(
            [f"{feat.type}: {feat.location}" for feat in record.features if feat.type != "source"]
        ),
        "Disulfide Bonds": ", ".join(
            [f"{feat.location}" for feat in record.features if "disulfide" in feat.type.lower()]
        ),
        "Catalytic Sites": ", ".join(
            [f"{feat.location}" for feat in record.features if "active site" in feat.type.lower()]
        ),
        "Authors": ", ".join([ref.authors for ref in record.annotations.get("references", []) if ref.authors]),
        "Journal": "; ".join([ref.journal for ref in record.annotations.get("references", []) if ref.journal]),
        "Title": "; ".join([ref.title for ref in record.annotations.get("references", []) if ref.title]),
        "Sequence": str(record.seq)
    }

    # Convert extracted data to DataFrame
    df = pd.DataFrame([parsed_data])

    return df


def calculate_sequence_identity(seq1, seq2):
    """
    Calculate percentage sequence identity between two sequences.
    Ignores gaps when computing identity.

    Args:
        seq1 (str): Reference sequence.
        seq2 (str): Target sequence to compare.

    Returns:
        float: Percentage sequence identity.
    """
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != '-' and b != '-')
    length = sum(1 for a, b in zip(seq1, seq2) if a != '-' and b != '-')

    if length == 0:
        return 0  # Avoid division by zero

    return (matches / length) * 100


def filter_and_save_sequences(input_fasta, reference_id=0, inequality='<', id_threshold=90, len_threshold=None, output_filtered_fasta=None, msa_dir='./', seq_dir='./'):
    """
    Filters an MSA file, removes sequences with >X% sequence identity to the reference,
    and saves the remaining sequences as a new FASTA file (without alignment) for MSA regeneration.

    Args:
        input_fasta (str): Path to input MSA file in FASTA format.
        reference_id (str): The identifier of the reference sequence.
        id_threshold (float): Sequence identity id_threshold (e.g., 90.0 for 90%).
        output_filtered_fasta (str): Path to output FASTA file with unaligned filtered sequences.
    """
    from Bio import SeqIO

    # get all records
    records = list(SeqIO.parse(msa_dir+input_fasta, "fasta"))
    num_records = len(records)

    # Find the reference sequence
    reference_seq = None
    if isinstance(reference_id,int):
        reference_seq = records[reference_id].seq
    else:
        for record in records:
            if record.id == reference_id:
                reference_seq = str(record.seq).replace("-", "")  # Remove gaps for sequence comparison
                break

    if reference_seq is None:
        print(f"Reference sequence '{reference_id}' not found in MSA.")
        sys.exit(1)
    else:
        reference_seq_len = len(reference_seq)
        seq_len_threshold = int(len_threshold*reference_seq_len)

    filtered_records = []
    for record in records:
        unaligned_seq = str(record.seq).replace("-", "")  # Remove gaps before saving
        seq_len = len(unaligned_seq)
        if record.id == reference_id:
            record.seq = Seq(reference_seq)
            filtered_records.append(record)  # Always keep reference
            continue

        seq_identity = calculate_sequence_identity(reference_seq, unaligned_seq)
        if eval(f'{seq_identity} {inequality} {id_threshold}'):
            print(f"Removed {record.id}: {seq_identity:.2f}% identity {inequality} {id_threshold}%")
        else:
            if len_threshold is None or seq_len > seq_len_threshold:
                record.seq = Seq(unaligned_seq)  # Remove gaps before saving
                filtered_records.append(record)
            else:
                print(f"Removed {record.id}: {seq_len} seq len < {seq_len_threshold}")

    # Save unaligned sequences for new MSA generation
    if output_filtered_fasta is None:
        output_filtered_fasta = input_fasta.replace('.fasta','_filt.fasta')
    SeqIO.write(filtered_records, seq_dir+output_filtered_fasta, "fasta")
    print(f"Filtered unaligned sequences saved to {seq_dir+output_filtered_fasta}. Kept {len(filtered_records)} sequences. >> Removed {num_records-len(filtered_records)} sequences.")

    return output_filtered_fasta


def run_clipkit(input_msa_fname, output_msa_fname, msa_dir, keep_ref_seq_positions=None, mode='kpic-gappy'):
    """
    Runs ClipKit on an input FASTA file and writes the trimmed MSA to an output file.

    Args:
        input_fasta (str): Path to the input MSA file in FASTA format.
        output_fasta (str): Path to the output trimmed MSA file (default: "output.fasta").
    """
    import subprocess
    # create auxiliary file to keep certain positions
    if keep_ref_seq_positions is not None:
        sequences, _, _ = fetch_sequences_from_fasta(msa_dir+input_msa_fname)
        ref_seq = sequences[keep_ref_seq_positions]
        pos_to_keep = [i+1 for i,aa in enumerate(ref_seq) if aa!='-']
        seq_to_keep = [aa for aa in ref_seq if aa!='-']
        print('Positions to keep:', len(pos_to_keep), pos_to_keep)
        print('Reference sequence:', ''.join(seq_to_keep))
        with open(msa_dir+'aux.txt','w') as f:
            for pos in pos_to_keep:
                f.write(f'{pos}\tkeep\n')
    try:
        # Run ClipKit
        if keep_ref_seq_positions is None:
            subprocess.run(["clipkit", msa_dir+input_msa_fname, "-o", msa_dir+output_msa_fname, "-m", mode], check=True)
        else:
            subprocess.run(["clipkit", msa_dir + input_msa_fname, "-o", msa_dir + output_msa_fname, "-a", msa_dir + 'aux.txt', "-m", 'cst'], check=True)
            # os.remove(msa_dir + 'aux.txt')
        print(f"ClipKit successfully trimmed MSA. Output saved to: {msa_dir+output_msa_fname}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ClipKit: {e}")

    return output_msa_fname


def replace_edge_gaps(input_fasta, output_fasta, msa_dir):
    """
    Replaces leading and trailing '-' (gaps) with 'X' in each sequence of an MSA file.

    Args:
        input_fasta (str): Path to the input MSA file in FASTA format.
        output_fasta (str): Path to the output modified MSA file.
    """
    modified_records = []

    for record in SeqIO.parse(msa_dir+input_fasta, "fasta"):
        seq_str = str(record.seq)

        # Replace leading and trailing '-' with 'X'
        modified_seq = list(seq_str)
        start, end = 0, len(modified_seq) - 1

        # Replace leading '-'
        while start < len(modified_seq) and modified_seq[start] == '-':
            modified_seq[start] = 'X'
            start += 1

        # Replace trailing '-'
        while end >= 0 and modified_seq[end] == '-':
            modified_seq[end] = 'X'
            end -= 1

        # Save the modified sequence
        record.seq = Seq("".join(modified_seq))
        modified_records.append(record)

    # Write the modified MSA to a new file
    SeqIO.write(modified_records, msa_dir+output_fasta, "fasta")
    print(f"Modified MSA saved to {msa_dir+output_fasta}")

    return output_fasta

