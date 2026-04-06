from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from abcode.tools.openprotein.openprotein_utils import connect_openprotein_session
from abcode.tools.utils.seq_utils import write_sequence_to_fasta


def create_openprotein_msa(
    *,
    seed_sequence: str,
    seed_sequence_name: Optional[str] = None,
    session: Optional[Any] = None,
    seq_fasta_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create an MSA job via OpenProtein API from a seed sequence or sequence set.

    Args:
        seed_sequence: Optional seed sequence for homolog-based MSA generation.
        sequences: Optional explicit sequence list. If multiple are provided they
            are joined with ":" and submitted as one MSA seed string.
        session: Optional authenticated OpenProtein session.
        wait: If true, block until MSA job finishes.

    Returns:
        Dict containing `msa_id`, `status`, `job`, and optional `msa_text`.
    """
    # start session and create MSA
    sess = session or connect_openprotein_session()

    # create MSA
    msa = sess.align.create_msa(seed_sequence.encode())

    # parse MSA
    r = msa.wait()

    # save to fasta
    if seq_fasta_path is not None:
        sequences = []
        seq_names = []
        sequences_degapped = []
        msa_iterator = msa.get()
        for (seq_name, seq) in msa_iterator:
            seq_name = seq_name.split('\t')[0]
            sequences.append(seq)
            seq_names.append(seq_name)
            seq_degapped = seq.replace('-','')
            sequences_degapped.append(seq_degapped)
        seq_names[0] = seed_sequence_name

        # save MSA to fasta
        write_sequence_to_fasta(sequences, seq_names, os.path.basename(seq_fasta_path).replace('sequences/','msa/'), os.path.dirname(seq_fasta_path) + '/')
        # save degapped sequences to fasta
        write_sequence_to_fasta(sequences_degapped, seq_names, os.path.basename(seq_fasta_path), os.path.dirname(seq_fasta_path)+'/')

    return msa


def upload_openprotein_msa(
        msa_fpath,
        session: Optional[Any] = None,
):
    # start session and create MSA
    sess = session or connect_openprotein_session()
    msa = session.align.upload_msa(msa_fpath.encode())
    return msa