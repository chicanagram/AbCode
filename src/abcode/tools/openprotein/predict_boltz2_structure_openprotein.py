#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import pandas as pd

# Allow direct execution of this file via an absolute path by adding the repo's
# package roots to sys.path before importing local packages.
SRC_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
for import_root in (str(SRC_ROOT), str(REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from abcode.tools.openprotein.openprotein_utils import connect_openprotein_session
from abcode.core.paths import setup_data_root
from abcode.tools.utils.struct_utils import convert_cif_to_pdb_pymol


def _chain_ids(n: int, start_index: int = 0) -> List[str]:
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    out: List[str] = []
    i = start_index
    while len(out) < n:
        if i < len(alphabet):
            out.append(alphabet[i])
        else:
            a = alphabet[(i // len(alphabet)) - 1]
            b = alphabet[i % len(alphabet)]
            out.append(f"{a}{b}")
        i += 1
    return out


def predict_boltz2(
    sequences: Sequence[str],
    smiles: Optional[Sequence[str]] = None,
    ccds: Optional[Sequence[str]] = None,
    *,
    session: Optional[Any] = None,
    use_single_sequence_mode: bool = False,
    templates: Optional[Sequence[Any]] = None,
    predict_affinity: bool = False,
    binder_chain: Optional[str] = None,
    wait: bool = True,
    return_arrays: bool = False,
    out_cif: Optional[Path] = None,
    out_summary_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Predict structure(s) with Boltz-2.
    - sequences: one or more protein sequences
    - smiles: optional ligand SMILES list
    - ccds: optional ligand CCD list
    """
    try:
        from openprotein.molecules import Complex, Ligand, Protein
    except ImportError as exc:
        raise RuntimeError("Missing dependency `openprotein`. Install it in your environment.") from exc

    if not sequences:
        raise ValueError("At least one protein sequence is required.")

    # Section: normalize inputs ------------------------------------------------
    smiles_list = list(smiles or [])
    ccd_list = list(ccds or [])
    if predict_affinity and not (smiles_list or ccd_list):
        raise ValueError("predict_affinity=True requires at least one ligand.")

    sess = session or connect_openprotein_session()
    print('Session:', sess)

    # Section: build molecular complex ----------------------------------------
    proteins = [Protein(sequence=s) for s in sequences]
    protein_chains = _chain_ids(len(proteins), start_index=0)

    ligands: List[Ligand] = [Ligand(smiles=s) for s in smiles_list] + [Ligand(ccd=c) for c in ccd_list]
    ligand_chains = _chain_ids(len(ligands), start_index=len(proteins))

    chain_map: Dict[str, Any] = {}
    for cid, p in zip(protein_chains, proteins):
        chain_map[cid] = p
    for cid, l in zip(ligand_chains, ligands):
        chain_map[cid] = l

    molecular_complex = Complex(chain_map)

    # Section: MSA setup -------------------------------------------------------
    if use_single_sequence_mode:
        for p in proteins:
            p.msa = Protein.single_sequence_mode
    else:
        msa_query = []
        for p in molecular_complex.get_proteins().values():
            msa_query.append(p.sequence)
        msa = sess.align.create_msa(seed=b":".join(msa_query))
        print('msa:', msa.id)

        for p in molecular_complex.get_proteins().values():
            p.msa = msa

    # Section: submit Boltz-2 job ---------------------------------------------
    fold_kwargs: Dict[str, Any] = {"sequences": [molecular_complex]}
    if templates:
        fold_kwargs["templates"] = list(templates)
    chosen_binder = binder_chain or (ligand_chains[0] if ligand_chains else None)
    if predict_affinity and chosen_binder:
        fold_kwargs["properties"] = [{"affinity": {"binder": chosen_binder}}]

    fold_job = sess.fold.boltz2.fold(**fold_kwargs)

    summary: Dict[str, Any] = {
        "job_id": getattr(fold_job, "job_id", ""),
        "protein_chains": protein_chains,
        "ligand_chains": ligand_chains,
        "predict_affinity": bool(predict_affinity),
        "binder_chain": chosen_binder,
    }

    # Section: async return (optional) ----------------------------------------
    if not wait:
        if out_summary_json:
            out_summary_json.parent.mkdir(parents=True, exist_ok=True)
            out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # Section: collect outputs -------------------------------------------------
    fold_job.wait_until_done(verbose=True)

    result = fold_job.get()
    structure = result[0]
    summary["status"] = str(getattr(fold_job, "status", ""))

    if out_cif:
        out_cif.parent.mkdir(parents=True, exist_ok=True)
        out_cif.write_text(structure.to_string(format="cif"), encoding="utf-8")
        summary["cif_path"] = str(out_cif.resolve())

    plddt = None
    try:
        plddt = fold_job.get_plddt()[0]
        summary["plddt_shape"] = list(getattr(plddt, "shape", []))
    except Exception as exc:
        summary["plddt_error"] = f"{type(exc).__name__}: {exc}"

    pae = None
    try:
        pae = fold_job.get_pae()[0]
        summary["pae_shape"] = list(getattr(pae, "shape", []))
    except Exception as exc:
        summary["pae_error"] = f"{type(exc).__name__}: {exc}"

    pde = None
    try:
        pde = fold_job.get_pde()[0]
        summary["pde_shape"] = list(getattr(pde, "shape", []))
    except Exception as exc:
        summary["pde_error"] = f"{type(exc).__name__}: {exc}"

    try:
        confidence = fold_job.get_confidence()[0]
        first = confidence[0] if isinstance(confidence, list) and confidence else confidence
        summary["confidence"] = first.model_dump() if hasattr(first, "model_dump") else str(first)
    except Exception as exc:
        summary["confidence_error"] = f"{type(exc).__name__}: {exc}"

    if predict_affinity:
        try:
            affinity = fold_job.get_affinity()[0]
            summary["affinity"] = affinity.model_dump() if hasattr(affinity, "model_dump") else str(affinity)
        except Exception as exc:
            summary["affinity_error"] = f"{type(exc).__name__}: {exc}"

    if out_summary_json:
        out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Runtime-only rich outputs (not serialized to JSON).
    if return_arrays:
        summary["runtime_plddt"] = plddt
        summary["runtime_pae"] = pae
        summary["runtime_pde"] = pde

    return summary


if __name__ == "__main__":

    # Edit these directly for local runs when you want file-backed inputs
    # without passing them on the CLI.
    data_root, _ = setup_data_root(
        root_key="biostream-developability-data",
        required_subfolders=("expdata", "pdb"),
        project_root=REPO_ROOT,
    )
    data_subfolder = "opensource"
    sequence_fasta: Optional[Path] = None
    sequence_csv: Optional[Path] = data_root / "expdata" / data_subfolder / f"{data_subfolder}.csv"
    sequence_cols = ["sequence_vh", "sequence_vl"]
    smiles_list = []
    use_single_sequence_mode = False
    predict_affinity = False
    binder_chain = None

    if sequence_csv is None:
        raise ValueError("Set `sequence_csv` or `sequence_fasta` under __main__ before running this script.")

    df = pd.read_csv(sequence_csv)
    output_dir = data_root / "pdb" / data_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)
    missed_struct_indices = []

    for i in range(2, len(df)):
        print(f'Processing structure {i}...')
        try:
            out_cif = output_dir / f"{i}.cif"
            sequences = df.iloc[i][sequence_cols].tolist()
            sequences = [str(s).strip() for s in sequences if pd.notna(s) and str(s).strip()]
            print(len(sequences), sequences)
            summary = predict_boltz2(
                sequences=sequences,
                smiles=smiles_list,
                use_single_sequence_mode=use_single_sequence_mode,
                predict_affinity=predict_affinity,
                binder_chain=binder_chain,
                wait=True,
                out_cif=out_cif,
                out_summary_json=out_cif.with_suffix(".json"),
            )
            convert_cif_to_pdb_pymol(str(out_cif), str(out_cif.with_suffix(".pdb")))
        except Exception as exc:
            print(exc, f">> Skipping structure {i}")
            missed_struct_indices.append(i)

    print('Missed structure indices:', missed_struct_indices)
