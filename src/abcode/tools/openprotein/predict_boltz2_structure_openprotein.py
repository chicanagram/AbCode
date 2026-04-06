#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from abcode.tools.openprotein.openprotein_utils import connect_openprotein_session


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


def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for standalone Boltz-2 prediction utility.

    Returns:
        Parsed argparse namespace with sequences, ligand options, and output paths.
    """
    p = argparse.ArgumentParser(description="Run Boltz-2 structure prediction via OpenProtein API.")
    p.add_argument("--sequence", action="append", default=[], help="Protein sequence. Repeat for multiple proteins.")
    p.add_argument("--smiles", action="append", default=[], help="Ligand SMILES. Repeat for multiple ligands.")
    p.add_argument("--ccd", action="append", default=[], help="Ligand CCD code. Repeat for multiple ligands.")
    p.add_argument("--single-sequence-mode", action="store_true", help="Use Protein.single_sequence_mode instead of MSA.")
    p.add_argument("--predict-affinity", action="store_true", help="Request affinity output.")
    p.add_argument("--binder-chain", default=None, help="Binder chain ID for affinity (defaults to first ligand chain).")
    p.add_argument("--no-wait", action="store_true", help="Submit job and return immediately.")
    p.add_argument("--out-cif", type=Path, default=Path("outputs/boltz2/boltz2_prediction.cif"))
    p.add_argument("--out-summary", type=Path, default=Path("outputs/boltz2/boltz2_summary.json"))
    return p.parse_args()


def main() -> None:
    """
    CLI entrypoint: run Boltz-2 prediction and print JSON summary.
    """
    args = _parse_args()
    summary = predict_boltz2(
        sequences=args.sequence,
        smiles=args.smiles,
        ccds=args.ccd,
        use_single_sequence_mode=args.single_sequence_mode,
        predict_affinity=args.predict_affinity,
        binder_chain=args.binder_chain,
        wait=not args.no_wait,
        out_cif=args.out_cif,
        out_summary_json=args.out_summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
