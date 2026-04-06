#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from abcode.tools.openprotein.openprotein_utils import connect_openprotein_session
from abcode.tools.utils.struct_utils import align_structures, convert_cif_to_pdb_pymol

def _chain_ids(n: int, *, used: Optional[set[str]] = None, start_index: int = 0) -> List[str]:
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    used_ids = set(used or set())
    out: List[str] = []
    i = start_index
    while len(out) < n:
        if i < len(alphabet):
            cid = alphabet[i]
        else:
            a = alphabet[(i // len(alphabet)) - 1]
            b = alphabet[i % len(alphabet)]
            cid = f"{a}{b}"
        if cid not in used_ids:
            out.append(cid)
            used_ids.add(cid)
        i += 1
    return out


def read_fasta_sequences(fasta_path: str | Path) -> List[str]:
    path = Path(fasta_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")
    sequences: List[str] = []
    current: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def read_smiles_strings(smiles_path: str | Path) -> List[str]:
    path = Path(smiles_path)
    if not path.exists():
        raise FileNotFoundError(f"SMILES file not found: {path}")
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def read_single_token_lines(path_like: str | Path) -> List[str]:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    out: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split()[0])
    return out


def _load_structure(pdb_path: str | Path | None = None, pdb_id: str | None = None) -> Any:
    try:
        from openprotein.molecules import Structure
    except ImportError as exc:
        raise RuntimeError("Missing dependency `openprotein`. Install it in your environment.") from exc

    if pdb_path:
        path = Path(pdb_path)
        if not path.exists():
            raise FileNotFoundError(f"PDB/mmCIF file not found: {path}")
        return Structure.from_filepath(str(path))
    if pdb_id:
        return Structure.from_pdb_id(pdb_id)
    return None


def _pick_single_sequence(
    *,
    direct_sequence: str | None = None,
    fasta_path: str | Path | None = None,
    label: str,
) -> Optional[str]:
    seq = (direct_sequence or "").strip()
    if seq:
        return seq
    if fasta_path:
        sequences = read_fasta_sequences(fasta_path)
        if not sequences:
            raise ValueError(f"No sequences found in {label} FASTA: {fasta_path}")
        return sequences[0]
    return None


def _to_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# Section: refold target helpers ------------------------------------------------
def _extract_refold_targets_from_design_spec(
    design_spec_path: str | Path | None,
) -> Dict[str, Any]:
    """
    Parse a BoltzGen design spec and extract optional refold context:
    ligand SMILES/CCDs and protein/peptide target hints.
    """
    out: Dict[str, Any] = {
        "ligand_smiles": [],
        "ligand_ccds": [],
        "target_sequence": None,
        "target_structure_path": None,
        "target_chain_id": None,
    }
    if not design_spec_path:
        return out
    spec = _load_design_spec(design_spec_path=design_spec_path)
    if not isinstance(spec, dict):
        return out
    entities = spec.get("entities", [])
    if not isinstance(entities, list):
        return out

    for entity in entities:
        if not isinstance(entity, dict):
            continue
        ligand_block = entity.get("ligand")
        if isinstance(ligand_block, dict):
            smiles = ligand_block.get("smiles")
            ccd = ligand_block.get("ccd")
            if isinstance(smiles, str) and smiles.strip():
                out["ligand_smiles"].append(smiles.strip())
            if isinstance(ccd, str) and ccd.strip():
                out["ligand_ccds"].append(ccd.strip())

        for key in ("protein", "peptide"):
            block = entity.get(key)
            if not isinstance(block, dict):
                continue
            if out["target_chain_id"] is None:
                chain_id = block.get("id")
                if isinstance(chain_id, str) and chain_id.strip():
                    out["target_chain_id"] = chain_id.strip()
            if out["target_structure_path"] is None:
                path_value = block.get("structure_path") or block.get("pdb_path")
                if isinstance(path_value, str) and path_value.strip():
                    out["target_structure_path"] = path_value.strip()
            if out["target_sequence"] is None:
                seq = block.get("sequence")
                if _is_explicit_protein_sequence(seq):
                    out["target_sequence"] = str(seq).strip()

    return out


def build_boltzgen_query(
    *,
    target_chain_id: str = "A",
    target_structure_path: str | Path | None = None,
    target_pdb_id: str | None = None,
    target_sequence: str | None = None,
    target_fasta_path: str | Path | None = None,
    target_is_cyclic: bool = False,
    design_chain_id: str = "B",
    design_structure_path: str | Path | None = None,
    design_pdb_id: str | None = None,
    design_source_chain_id: str | None = None,
    design_sequence: str | None = None,
    design_fasta_path: str | Path | None = None,
    design_is_cyclic: bool = False,
    design_group_id: int | None = 1,
    ligand_smiles: Optional[Sequence[str]] = None,
    ligand_smiles_path: str | Path | None = None,
    ligand_ccds: Optional[Sequence[str]] = None,
    ligand_ccd_path: str | Path | None = None,
    ligand_chain_start_index: int = 3,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build an OpenProtein Complex query for BoltzGen.

    Inputs can come from structure files/IDs, FASTA, and/or direct sequence/SMILES.
    """
    try:
        from openprotein.molecules import Complex, Ligand, Protein
    except ImportError as exc:
        raise RuntimeError("Missing dependency `openprotein`. Install it in your environment.") from exc

    query_metadata: Dict[str, Any] = {}
    chain_map: Dict[str, Any] = {}

    target_structure = _load_structure(target_structure_path, target_pdb_id)
    if target_structure is not None:
        first_complex = target_structure[0]
        target = first_complex.get_protein(chain_id=target_chain_id)
        query_metadata["target_source"] = "structure"
        chain_map[target_chain_id] = target
    else:
        seq = _pick_single_sequence(
            direct_sequence=target_sequence,
            fasta_path=target_fasta_path,
            label="target",
        )
        if seq:
            target = Protein(sequence=seq)
            if target_is_cyclic:
                target.cyclic = True
            query_metadata["target_source"] = "sequence"
            chain_map[target_chain_id] = target
        else:
            query_metadata["target_source"] = "none"

    design_structure = _load_structure(design_structure_path, design_pdb_id)
    if design_structure is not None:
        source_chain = design_source_chain_id or design_chain_id
        first_complex = design_structure[0]
        design = first_complex.get_protein(chain_id=source_chain)
        query_metadata["design_source"] = "structure"
    else:
        seq = _pick_single_sequence(
            direct_sequence=design_sequence,
            fasta_path=design_fasta_path,
            label="design",
        )
        if seq:
            design = Protein(sequence=seq)
            if design_is_cyclic:
                design.cyclic = True
            query_metadata["design_source"] = "sequence"
        else:
            design = None
            query_metadata["design_source"] = "none"

    if design is not None:
        if design_group_id is not None:
            design = design.set_group(design_group_id)
        chain_map[design_chain_id] = design

    smiles_list = list(ligand_smiles or [])
    if ligand_smiles_path:
        smiles_list.extend(read_smiles_strings(ligand_smiles_path))
    ccd_list = [c.strip() for c in list(ligand_ccds or []) if str(c).strip()]
    if ligand_ccd_path:
        ccd_list.extend(read_single_token_lines(ligand_ccd_path))

    ligand_specs: List[Tuple[str, str]] = [("smiles", s) for s in smiles_list] + [("ccd", c) for c in ccd_list]
    if ligand_specs:
        ligand_chain_ids = _chain_ids(
            len(ligand_specs),
            used=set(chain_map.keys()),
            start_index=ligand_chain_start_index,
        )
        for chain_id, (kind, value) in zip(ligand_chain_ids, ligand_specs):
            if kind == "smiles":
                chain_map[chain_id] = Ligand(smiles=value)
            else:
                chain_map[chain_id] = Ligand(ccd=value)
        query_metadata["ligand_chain_ids"] = ligand_chain_ids
        query_metadata["num_ligands"] = len(ligand_specs)
    else:
        query_metadata["ligand_chain_ids"] = []
        query_metadata["num_ligands"] = 0

    if not chain_map:
        raise ValueError(
            "No query entities provided. Add at least one protein/peptide target, design protein, or ligand."
        )

    query = Complex(chain_map)
    query_metadata["query_chain_ids"] = list(chain_map.keys())
    return query, query_metadata


def _load_design_spec(
    design_spec: Optional[Dict[str, Any]] = None,
    design_spec_path: str | Path | None = None,
) -> Optional[Dict[str, Any]]:
    if design_spec is not None:
        return design_spec
    if design_spec_path is None:
        return None
    path = Path(design_spec_path)
    if not path.exists():
        raise FileNotFoundError(f"Design spec file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML design specs require `pyyaml`. Install it or pass JSON."
        ) from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Design spec must parse to a dictionary.")
    return loaded


_AA_RE = re.compile(r"^[A-Za-z\*\-]+$")


def _is_explicit_protein_sequence(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    # Avoid treating range-like tokens (e.g. "140..180") as raw sequences.
    if ".." in text:
        return False
    return bool(_AA_RE.match(text))


def validate_design_with_boltzgen_kwargs(
    kwargs: Dict[str, Any],
    *,
    check_paths: bool = True,
) -> Dict[str, Any]:
    """
    Validate kwargs intended for `design_with_boltzgen(**kwargs)`.
    Returns a report dict: {"ok": bool, "errors": [...], "warnings": [...]}.
    """
    errors: List[str] = []
    warnings: List[str] = []

    has_target_protein = any(
        str(kwargs.get(k) or "").strip()
        for k in ("target_structure_path", "target_pdb_id")
    )
    has_target_peptide = any(
        str(kwargs.get(k) or "").strip()
        for k in ("target_sequence", "target_fasta_path")
    )
    ligand_smiles = kwargs.get("ligand_smiles")
    ligand_ccds = kwargs.get("ligand_ccds")
    has_ligand = bool(ligand_smiles) or bool(ligand_ccds) or bool(kwargs.get("ligand_smiles_path")) or bool(kwargs.get("ligand_ccd_path"))
    has_design_spec = bool(kwargs.get("design_spec") or kwargs.get("design_spec_path"))

    if not (has_target_protein or has_target_peptide or has_ligand or has_design_spec):
        errors.append(
            "Missing target specification. Provide at least one target modality: "
            "ligand (`ligand_smiles`/`ligand_ccds`), peptide (`target_sequence`/`target_fasta_path`), "
            "or protein structure (`target_structure_path`/`target_pdb_id`)."
        )

    target_sequence = kwargs.get("target_sequence")
    if isinstance(target_sequence, str) and target_sequence.strip():
        if not _is_explicit_protein_sequence(target_sequence):
            errors.append(
                "Invalid `target_sequence`. Provide amino-acid letters only "
                "(range placeholders like `140..180` are not valid here)."
            )

    design_sequence = kwargs.get("design_sequence")
    if isinstance(design_sequence, str) and design_sequence.strip():
        if not _is_explicit_protein_sequence(design_sequence):
            errors.append(
                "Invalid `design_sequence`. Provide amino-acid letters only."
            )

    try:
        n_structures = int(kwargs.get("n_structures", 0))
        if n_structures < 1:
            errors.append("`n_structures` must be >= 1.")
    except Exception:
        errors.append("`n_structures` must be an integer >= 1.")

    if check_paths:
        path_keys = [
            "target_structure_path",
            "target_fasta_path",
            "design_structure_path",
            "design_fasta_path",
            "ligand_smiles_path",
            "design_spec_path",
        ]
        for key in path_keys:
            value = kwargs.get(key)
            if not value:
                continue
            path = Path(value)
            if not path.exists():
                errors.append(f"Path does not exist for `{key}`: {path}")

    if not has_ligand and not has_design_spec:
        warnings.append("No ligand target provided (`ligand_smiles`/`ligand_ccds` are empty).")
    if has_design_spec and not (has_target_protein or has_target_peptide or has_ligand):
        warnings.append("Target may be defined inside `design_spec`; skipping strict target field enforcement.")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def assert_valid_design_with_boltzgen_kwargs(
    kwargs: Dict[str, Any],
    *,
    check_paths: bool = True,
) -> None:
    report = validate_design_with_boltzgen_kwargs(kwargs, check_paths=check_paths)
    if report["ok"]:
        return
    lines = ["Invalid BoltzGen inputs:"] + [f"- {msg}" for msg in report["errors"]]
    raise ValueError("\n".join(lines))


def boltzgen_yaml_to_design_kwargs(
    yaml_path: str | Path,
    *,
    include_design_spec_path: bool = True,
    include_design_spec_object: bool = False,
    include_query_fields: bool = False,
) -> Dict[str, Any]:
    """
    Convert a BoltzGen-style YAML file into kwargs for `design_with_boltzgen`.

    Mapping strategy (when include_query_fields=True):
    - First protein entity -> target input
    - Second protein entity -> design input (optional)
    - Ligand entities with `smiles` -> `ligand_smiles`
    - Keeps IDs as chain IDs where available.

    By default (`include_query_fields=False`), only design-spec references are returned.
    This avoids chain-ID mismatches between auto-built query chains and YAML entity IDs.
    """
    spec = _load_design_spec(design_spec_path=yaml_path)
    if not isinstance(spec, dict):
        raise ValueError("YAML spec must parse to a dictionary.")

    kwargs: Dict[str, Any] = {}
    entities = spec.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("YAML spec field `entities` must be a list if present.")

    proteins: List[Dict[str, Any]] = []
    ligand_smiles: List[str] = []
    ligand_ccds: List[str] = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        protein_block = entity.get("protein")
        if isinstance(protein_block, dict):
            proteins.append(protein_block)
        ligand_block = entity.get("ligand")
        if isinstance(ligand_block, dict):
            smiles = ligand_block.get("smiles")
            if isinstance(smiles, str) and smiles.strip():
                ligand_smiles.append(smiles.strip())
            ccd = ligand_block.get("ccd")
            if isinstance(ccd, str) and ccd.strip():
                ligand_ccds.append(ccd.strip())
        peptide_block = entity.get("peptide")
        if isinstance(peptide_block, dict) and not proteins:
            proteins.append(peptide_block)

    if include_query_fields and proteins:
        p0 = proteins[0]
        chain = p0.get("id")
        if isinstance(chain, str) and chain.strip():
            kwargs["target_chain_id"] = chain.strip()
        seq = p0.get("sequence")
        if _is_explicit_protein_sequence(seq):
            kwargs["target_sequence"] = str(seq).strip()
        elif isinstance(seq, str):
            # Keep range/templated sequence in design_spec only.
            pass
        pdb_id = p0.get("pdb_id")
        if isinstance(pdb_id, str) and pdb_id.strip():
            kwargs["target_pdb_id"] = pdb_id.strip()
        struct_path = p0.get("structure_path") or p0.get("pdb_path")
        if isinstance(struct_path, str) and struct_path.strip():
            kwargs["target_structure_path"] = struct_path.strip()

    if include_query_fields and len(proteins) >= 2:
        p1 = proteins[1]
        chain = p1.get("id")
        if isinstance(chain, str) and chain.strip():
            kwargs["design_chain_id"] = chain.strip()
        seq = p1.get("sequence")
        if _is_explicit_protein_sequence(seq):
            kwargs["design_sequence"] = str(seq).strip()
        pdb_id = p1.get("pdb_id")
        if isinstance(pdb_id, str) and pdb_id.strip():
            kwargs["design_pdb_id"] = pdb_id.strip()
        struct_path = p1.get("structure_path") or p1.get("pdb_path")
        if isinstance(struct_path, str) and struct_path.strip():
            kwargs["design_structure_path"] = struct_path.strip()
        group = p1.get("group")
        if isinstance(group, int):
            kwargs["design_group_id"] = group

    if include_query_fields and ligand_smiles:
        kwargs["ligand_smiles"] = ligand_smiles
    if include_query_fields and ligand_ccds:
        kwargs["ligand_ccds"] = ligand_ccds

    if include_design_spec_path:
        kwargs["design_spec_path"] = Path(yaml_path)
    if include_design_spec_object:
        kwargs["design_spec"] = spec
        kwargs.pop("design_spec_path", None)

    return kwargs


def design_with_boltzgen_yaml(
    yaml_path: str | Path,
    **overrides: Any,
) -> Dict[str, Any]:
    """
    Run `design_with_boltzgen` using kwargs derived from a YAML design spec.
    `overrides` can replace any parsed/default field.
    """
    kwargs = boltzgen_yaml_to_design_kwargs(yaml_path)
    kwargs.update(overrides)
    return design_with_boltzgen(**kwargs)


def run_boltzgen_design(
    query: Any = None,
    *,
    design_spec: Optional[Dict[str, Any]] = None,
    design_spec_path: str | Path | None = None,
    n_structures: int = 16,
    diffusion_batch_size: Optional[int] = None,
    step_scale: Optional[float] = None,
    noise_scale: Optional[float] = None,
    wait: bool = True,
    verbose: bool = True,
    out_dir: str | Path | None = None,
    out_summary_json: str | Path | None = None,
    output_prefix: str = "boltzgen_design",
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Submit and collect BoltzGen de novo design jobs.

    Args:
        query: Optional OpenProtein query object (if already constructed).
        design_spec: Optional in-memory design-spec dictionary.
        design_spec_path: Optional YAML design-spec path.
        n_structures: Number of structures to generate.
        diffusion_batch_size: Optional BoltzGen batch size override.
        step_scale: Optional diffusion step-scale override.
        noise_scale: Optional diffusion noise-scale override.
        wait: Whether to block until job completion.
        verbose: Verbose wait logging.
        out_dir: Optional directory to save generated CIF/PDB structures.
        out_summary_json: Optional summary JSON output path.
        output_prefix: Output filename prefix.
        session: Optional existing OpenProtein session.

    Returns:
        Dict with job handle, summary metadata, and generated structures (if waited).
    """
    sess = session or connect_openprotein_session()
    resolved_design_spec = _load_design_spec(design_spec, design_spec_path)
    print('resolved_design_spec:', resolved_design_spec)

    kwargs: Dict[str, Any] = {"N": n_structures}
    if query is not None:
        kwargs["query"] = query
    if resolved_design_spec is not None:
        kwargs["design_spec"] = resolved_design_spec
    if diffusion_batch_size is not None:
        kwargs["diffusion_batch_size"] = diffusion_batch_size
    if step_scale is not None:
        kwargs["step_scale"] = step_scale
    if noise_scale is not None:
        kwargs["noise_scale"] = noise_scale

    boltzgen_job = sess.models.boltzgen.generate(**kwargs)
    summary: Dict[str, Any] = {
        "job_id": str(getattr(boltzgen_job, "job_id", "")),
        "status": str(getattr(boltzgen_job, "status", "")),
        "n_structures_requested": n_structures,
    }

    if not wait:
        if out_summary_json:
            out_path = Path(out_summary_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"job": boltzgen_job, "summary": summary, "structures": None}

    boltzgen_job.wait_until_done(verbose=verbose)
    generated_structures = boltzgen_job.get()

    summary["status"] = str(getattr(boltzgen_job, "status", ""))
    summary["n_structures_returned"] = len(generated_structures)
    summary["structure_paths"] = []

    if out_dir:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = str(output_prefix).strip() or "boltzgen_design"
        for i, structure in enumerate(generated_structures):
            out_path = output_dir / f"{safe_prefix}_{i:03d}.cif"
            out_path.write_text(structure.to_string(format="cif"), encoding="utf-8")
            # convert cif to pdb
            convert_cif_to_pdb_pymol(out_path, str(out_path).replace('cif','pdb'))

    if out_summary_json:
        out_path = Path(out_summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {"job": boltzgen_job, "summary": summary, "structures": generated_structures}


def run_proteinmpnn_from_structures(
    generated_structures: Sequence[Any],
    *,
    target_chain_id: str = "A",
    design_chain_id: str = "B",
    scaffold_sequence: str | None = None,
    num_samples: int = 8,
    temperature: float = 0.1,
    seed: Optional[int] = 42,
    linker: str = "GGGGSGGGGSGGGGS",
    wait: bool = True,
    out_csv: str | Path | None = None,
    session: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Run ProteinMPNN sequence design on generated structures.

    Args:
        generated_structures: Structure objects to design against.
        target_chain_id: Optional target chain used as context.
        design_chain_id: Chain to redesign.
        scaffold_sequence: Optional fixed scaffold sequence for design chain.
        num_samples: Number of sampled sequences per structure.
        temperature: ProteinMPNN sampling temperature.
        seed: Optional random seed.
        linker: Linker used when concatenating target/design chains.
        wait: Whether to wait for all jobs to finish.
        out_csv: Optional output CSV path for designed sequences.
        session: Optional existing OpenProtein session.

    Returns:
        List of sequence-record dicts (structure index, sequence index, score, sequence).
    """
    sess = session or connect_openprotein_session()

    jobs: List[Any] = []
    binder_lengths: List[int] = []

    for structure in generated_structures:
        # Design chain must be a protein chain.
        binder = structure.get_protein(chain_id=design_chain_id).copy()
        if scaffold_sequence:
            binder = binder.set_sequence(scaffold_sequence)
        binder = binder.mask_structure(side_chain_only=True)
        binder_lengths.append(len(_to_text(binder.sequence)))

        # Target chain may be absent or non-protein (for example ligand-only target runs).
        target = None
        if target_chain_id and target_chain_id != design_chain_id:
            try:
                target = structure.get_protein(chain_id=target_chain_id)
            except Exception:
                target = None

        if target is None:
            inverse_folding_input = binder
        else:
            inverse_folding_input = target + linker + binder

        job = sess.models.proteinmpnn.generate(
            query=inverse_folding_input,
            num_samples=num_samples,
            temperature=temperature,
            seed=seed,
        )
        jobs.append(job)

    if wait:
        for job in jobs:
            job.wait_until_done()

    records: List[Dict[str, Any]] = []
    for structure_idx, (job, binder_length) in enumerate(zip(jobs, binder_lengths)):
        results = job.get()
        for sequence_idx, proteinmpnn_result in enumerate(results):
            full_sequence = _to_text(proteinmpnn_result.sequence)
            designed_sequence = full_sequence[-binder_length:]
            score_attr = getattr(proteinmpnn_result, "score", None)
            try:
                score_value = float(score_attr.mean().item()) if score_attr is not None else None
            except Exception:
                score_value = None
            records.append(
                {
                    "structure_idx": structure_idx,
                    "sequence_idx": sequence_idx,
                    "score": score_value,
                    "sequence": designed_sequence,
                }
            )

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["structure_idx", "sequence_idx", "score", "sequence"]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    return records


def run_proteinmpnn_postdesign_pipeline(
    generated_structures: Sequence[Any],
    *,
    out_dir: str | Path,
    output_tag: str = "",
    target_chain_id: str = "A",
    design_chain_id: str = "B",
    scaffold_sequence: str | None = None,
    num_samples: int = 8,
    temperature: float = 0.1,
    seed: Optional[int] = 42,
    run_boltz2_refold: bool = False,
    refold_target_mode: str = "auto",
    refold_ligand_smiles: Optional[Sequence[str]] = None,
    refold_ligand_ccds: Optional[Sequence[str]] = None,
    refold_target_structure_path: str | Path | None = None,
    refold_target_structure_chain_id: str = "A",
    design_spec_path: str | Path | None = None,
    auto_refold_targets_from_design_spec: bool = True,
    align_refolds_to_generated: bool = True,
    compute_metrics: bool = False,
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Post-BoltzGen pipeline:
    1) Generate sequences with ProteinMPNN
    2) Optionally refold each target+designed complex with Boltz-2
    3) Optionally align Boltz-2 refolds to generated structures
    4) Optionally compute structural metrics
    """
    try:
        import pandas as pd
        from openprotein.molecules import Structure as OPStructure
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for postdesign pipeline.") from exc

    from abcode.tools.openprotein.predict_boltz2_structure_openprotein import predict_boltz2

    # Section: output paths ----------------------------------------------------
    output_dir = Path(out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(output_tag).strip()
    mpnn_csv_name = f"{tag}_proteinmpnn_sequences.csv" if tag else "proteinmpnn_sequences.csv"
    mpnn_csv = output_dir / mpnn_csv_name

    records = run_proteinmpnn_from_structures(
        generated_structures=generated_structures,
        target_chain_id=target_chain_id,
        design_chain_id=design_chain_id,
        scaffold_sequence=scaffold_sequence,
        num_samples=num_samples,
        temperature=temperature,
        seed=seed,
        out_csv=mpnn_csv,
        session=session,
    )
    mpnn_df = pd.DataFrame.from_records(records).sort_values(["structure_idx", "sequence_idx"]).reset_index(drop=True)
    if "design_idx" not in mpnn_df.columns:
        mpnn_df["design_idx"] = range(len(mpnn_df))
    mpnn_records_csv = output_dir / (f"{tag}_proteinmpnn_records.csv" if tag else "proteinmpnn_records.csv")
    mpnn_df.to_csv(mpnn_records_csv, index=False)

    payload: Dict[str, Any] = {
        "mpnn_records": records,
        "mpnn_df": mpnn_df,
        "mpnn_csv": str(mpnn_csv),
        "mpnn_records_csv": str(mpnn_records_csv),
    }

    if not run_boltz2_refold:
        return payload

    # Section: resolve refold target context ----------------------------------
    refold_dir = output_dir / "boltz2_refolds"
    refold_dir.mkdir(parents=True, exist_ok=True)
    target_mode = str(refold_target_mode).strip().lower()
    ligand_smiles_resolved = list(refold_ligand_smiles or [])
    ligand_ccds_resolved = list(refold_ligand_ccds or [])
    spec_target_sequence: Optional[str] = None
    spec_target_structure_path: Optional[str] = None
    spec_target_chain_id: Optional[str] = None

    if auto_refold_targets_from_design_spec and design_spec_path:
        parsed = _extract_refold_targets_from_design_spec(design_spec_path)
        if not ligand_smiles_resolved:
            ligand_smiles_resolved = list(parsed.get("ligand_smiles", []))
        if not ligand_ccds_resolved:
            ligand_ccds_resolved = list(parsed.get("ligand_ccds", []))
        spec_target_sequence = parsed.get("target_sequence")
        spec_target_structure_path = parsed.get("target_structure_path")
        spec_target_chain_id = parsed.get("target_chain_id")

    # Optional target template from explicit or design-spec structure path.
    target_template = None
    target_template_sequence: Optional[str] = None
    target_structure_candidate = (
        str(refold_target_structure_path).strip()
        if refold_target_structure_path
        else str(spec_target_structure_path or "").strip()
    )
    if target_structure_candidate:
        try:
            from openprotein.molecules import Structure as OPStructure

            s = OPStructure.from_filepath(target_structure_candidate)
            c = s[0]
            chain_choice = str(refold_target_structure_chain_id or spec_target_chain_id or target_chain_id or "A")
            target_template = c.get_protein(chain_id=chain_choice)
            seq = target_template.sequence
            target_template_sequence = seq.decode() if isinstance(seq, bytes) else str(seq)
        except Exception:
            target_template = None
            target_template_sequence = None

    # Section: refold loop -----------------------------------------------------
    predicted_structures: List[Any] = []
    predicted_paes: List[Any] = []
    refold_summaries: List[Dict[str, Any]] = []
    generated_ref_pdb_cache: Dict[int, Path] = {}
    for _, row in mpnn_df.iterrows():
        s_idx = int(row["structure_idx"])
        q_idx = int(row["sequence_idx"])
        designed_sequence = str(row["sequence"]).strip()
        generated_structure = generated_structures[s_idx]

        target_sequence: Optional[str] = None
        if target_mode in {"auto", "protein", "peptide"}:
            # Prefer explicit target template sequence, then generated structure target, then design-spec sequence.
            if target_template_sequence:
                target_sequence = target_template_sequence
            else:
                try:
                    seq = generated_structure.get_protein(chain_id=target_chain_id).sequence
                    target_sequence = seq.decode() if isinstance(seq, bytes) else str(seq)
                except Exception:
                    target_sequence = spec_target_sequence

        # Keep designed binder as the first protein chain (Chain A for 2-protein refolds).
        sequences = [designed_sequence] if not target_sequence else [designed_sequence, target_sequence]
        smiles = list(ligand_smiles_resolved or []) if target_mode in {"auto", "ligand"} else []
        ccds = list(ligand_ccds_resolved or []) if target_mode in {"auto", "ligand"} else []

        out_cif = refold_dir / f"boltz2_refold_s{s_idx:03d}_q{q_idx:03d}.cif"
        out_json = refold_dir / f"boltz2_refold_s{s_idx:03d}_q{q_idx:03d}.json"
        templates = [target_template] if target_template is not None else None

        summary = predict_boltz2(
            sequences=sequences,
            smiles=smiles,
            ccds=ccds,
            use_single_sequence_mode=True,
            templates=templates,
            predict_affinity=False,
            wait=True,
            return_arrays=True,
            out_cif=out_cif,
            out_summary_json=out_json,
            session=session,
        )
        # Section: align predicted structure to original generated structure ----
        out_pdb = out_cif.with_suffix(".pdb")
        convert_cif_to_pdb_pymol(out_cif, out_pdb)

        aligned_ok = False
        if align_refolds_to_generated:
            generated_ref_pdb = generated_ref_pdb_cache.get(s_idx)
            if generated_ref_pdb is None:
                generated_ref_cif = refold_dir / f"generated_ref_s{s_idx:03d}.cif"
                generated_ref_pdb = refold_dir / f"generated_ref_s{s_idx:03d}.pdb"
                generated_ref_cif.write_text(generated_structure.to_string(format="cif"), encoding="utf-8")
                convert_cif_to_pdb_pymol(generated_ref_cif, generated_ref_pdb)
                generated_ref_pdb_cache[s_idx] = generated_ref_pdb
            try:
                align_structures(str(out_pdb), str(generated_ref_pdb))
                aligned_ok = True
            except Exception:
                aligned_ok = False

        try:
            predicted = OPStructure.from_filepath(str(out_pdb))[0]
        except Exception:
            predicted = OPStructure.from_filepath(str(out_cif))[0]
        predicted_structures.append(predicted)
        predicted_paes.append(summary.get("runtime_pae"))
        refold_summaries.append(
            {
                "structure_idx": s_idx,
                "sequence_idx": q_idx,
                "out_cif": str(out_cif),
                "out_pdb": str(out_pdb),
                "out_json": str(out_json),
                "status": summary.get("status", ""),
                "aligned_to_generated": aligned_ok,
                "ligand_smiles_count": len(smiles),
                "ligand_ccd_count": len(ccds),
            }
        )

    payload["predicted_structures"] = predicted_structures
    refold_summaries_df = pd.DataFrame(refold_summaries)
    refold_summary_csv = output_dir / (
        f"{tag}_proteinmpnn_refold_summary.csv" if tag else "proteinmpnn_refold_summary.csv"
    )
    refold_summaries_df.to_csv(refold_summary_csv, index=False)
    payload["refold_summaries_df"] = refold_summaries_df
    payload["refold_summaries_csv"] = str(refold_summary_csv)

    if compute_metrics:
        metrics_df = evaluate_boltz2_refold_metrics(
            generated_structures=generated_structures,
            predicted_structures=predicted_structures,
            sequence_records=mpnn_df,
            target_chain_id=target_chain_id,
            design_chain_id=design_chain_id,
            predicted_paes=predicted_paes,
        )
        metrics_csv = output_dir / (
            f"{tag}_proteinmpnn_predicted_structure_metrics.csv"
            if tag
            else "proteinmpnn_predicted_structure_metrics.csv"
        )
        metrics_df.reset_index().to_csv(metrics_csv, index=False)
        payload["metrics_df"] = metrics_df
        payload["metrics_csv"] = str(metrics_csv)
    return payload


def evaluate_boltz2_refold_metrics(
    generated_structures: Sequence[Any],
    predicted_structures: Sequence[Any],
    sequence_records: Any,
    *,
    target_chain_id: str = "A",
    design_chain_id: str = "B",
    predicted_paes: Optional[Sequence[Any]] = None,
) -> Any:
    """
    Compare original BoltzGen structures vs Boltz-2 refolds from ProteinMPNN sequences.

    Returns a pandas DataFrame indexed by (structure_idx, sequence_idx).
    `predicted_paes` is optional; if omitted, `ipae` will be NaN.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("This helper requires numpy and pandas.") from exc

    if hasattr(sequence_records, "iterrows"):
        records_df = sequence_records.copy()
    else:
        records_df = pd.DataFrame.from_records(list(sequence_records))

    if "structure_idx" not in records_df.columns or "sequence_idx" not in records_df.columns:
        raise ValueError("sequence_records must include `structure_idx` and `sequence_idx`.")

    records: List[Dict[str, Any]] = []
    for linear_idx, row in records_df.reset_index(drop=True).iterrows():
        s_idx = int(row["structure_idx"])
        q_idx = int(row["sequence_idx"])
        generated_structure = generated_structures[s_idx]
        predicted_structure = predicted_structures[linear_idx]

        # Section: chain-level metrics -----------------------------------------
        try:
            generated_binder = generated_structure.get_protein(chain_id=design_chain_id)
            predicted_binder = predicted_structure.get_protein(chain_id=design_chain_id)
            binder_rmsd = float(predicted_binder.rmsd(generated_binder))
            binder_plddt = float(predicted_binder.plddt.mean())
        except Exception:
            binder_rmsd = float("nan")
            binder_plddt = float("nan")

        # Section: overall RMSD (with fallback) -------------------------------
        try:
            rmsd = float(predicted_structure.rmsd(generated_structure))
        except Exception:
            # Fallback when full-complex chain composition differs.
            rmsd = binder_rmsd if not np.isnan(binder_rmsd) else float("nan")

        # Section: iPAE --------------------------------------------------------
        ipae = float("nan")
        if predicted_paes is not None and linear_idx < len(predicted_paes):
            try:
                pae = np.asarray(predicted_paes[linear_idx]).squeeze(0)
                target_len = None
                try:
                    target_len = len(generated_structure.get_protein(chain_id=target_chain_id))
                except Exception:
                    target_len = None
                if target_len is None:
                    try:
                        target_len = len(generated_structure.get_protein(chain_id=design_chain_id))
                    except Exception:
                        target_len = None
                if target_len and target_len < pae.shape[0]:
                    ipae0 = float(np.mean(pae[:target_len, target_len:]))
                    ipae1 = float(np.mean(pae[target_len:, :target_len]))
                    ipae = (ipae0 + ipae1) / 2.0
            except Exception:
                ipae = float("nan")

        records.append(
            {
                "design_idx": row.get("design_idx", linear_idx),
                "structure_idx": s_idx,
                "sequence_idx": q_idx,
                "rmsd": rmsd,
                "ipae": ipae,
                "binder_rmsd": binder_rmsd,
                "binder_plddt": binder_plddt,
                "score": row.get("score", float("nan")),
                "sequence": row.get("sequence", ""),
            }
        )

    return pd.DataFrame.from_records(records).set_index(["structure_idx", "sequence_idx"])


def filter_and_select_designs(
    metrics_df: Any,
    *,
    rmsd_max: float = 2.5,
    ipae_max: float = 10.0,
    binder_rmsd_max: float = 1.0,
    binder_plddt_min: float = 80.0,
    rank_by: str = "ipae",
) -> Tuple[Any, Any]:
    """
    Filter candidate designs by thresholds, then keep the best sequence per structure.
    Returns (df_filtered, df_selected).
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("This helper requires pandas.") from exc

    df = metrics_df.copy() if hasattr(metrics_df, "copy") else pd.DataFrame(metrics_df)
    required = {"rmsd", "ipae", "binder_rmsd", "binder_plddt"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"metrics_df missing columns: {sorted(missing)}")

    df_filtered = df[
        (df["rmsd"] < rmsd_max)
        & (df["ipae"] < ipae_max)
        & (df["binder_rmsd"] < binder_rmsd_max)
        & (df["binder_plddt"] > binder_plddt_min)
    ]

    if df_filtered.empty:
        return df_filtered, df_filtered

    df_selected = (
        df_filtered.reset_index()
        .sort_values(by=rank_by)
        .groupby("structure_idx", sort=False)
        .first()
        .reset_index()
        .set_index(["structure_idx", "sequence_idx"])
    )
    return df_filtered, df_selected


def design_with_boltzgen(
    *,
    target_chain_id: str = "A",
    target_structure_path: str | Path | None = None,
    target_pdb_id: str | None = None,
    target_sequence: str | None = None,
    target_fasta_path: str | Path | None = None,
    target_is_cyclic: bool = False,
    design_chain_id: str = "B",
    design_structure_path: str | Path | None = None,
    design_pdb_id: str | None = None,
    design_source_chain_id: str | None = None,
    design_sequence: str | None = None,
    design_fasta_path: str | Path | None = None,
    design_is_cyclic: bool = False,
    design_group_id: int | None = 1,
    ligand_smiles: Optional[Sequence[str]] = None,
    ligand_smiles_path: str | Path | None = None,
    ligand_ccds: Optional[Sequence[str]] = None,
    ligand_ccd_path: str | Path | None = None,
    design_spec: Optional[Dict[str, Any]] = None,
    design_spec_path: str | Path | None = None,
    n_structures: int = 16,
    out_dir: str | Path | None = None,
    out_summary_json: str | Path | None = None,
    output_prefix: str = "boltzgen_design",
    session: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    High-level end-to-end de novo design orchestration around BoltzGen.

    Input format:
        Keyword arguments defining protein/ligand inputs, design constraints,
        OpenProtein runtime controls, and output paths.

    Returns:
        Dict containing generated structures, metadata, and optional downstream artifacts.
    """
    has_explicit_query_inputs = any(
        [
            bool(target_structure_path),
            bool(target_pdb_id),
            bool(target_sequence),
            bool(target_fasta_path),
            bool(design_structure_path),
            bool(design_pdb_id),
            bool(design_sequence),
            bool(design_fasta_path),
            bool(ligand_smiles),
            bool(ligand_smiles_path),
            bool(ligand_ccds),
            bool(ligand_ccd_path),
        ]
    )

    if has_explicit_query_inputs:
        query, query_metadata = build_boltzgen_query(
            target_chain_id=target_chain_id,
            target_structure_path=target_structure_path,
            target_pdb_id=target_pdb_id,
            target_sequence=target_sequence,
            target_fasta_path=target_fasta_path,
            target_is_cyclic=target_is_cyclic,
            design_chain_id=design_chain_id,
            design_structure_path=design_structure_path,
            design_pdb_id=design_pdb_id,
            design_source_chain_id=design_source_chain_id,
            design_sequence=design_sequence,
            design_fasta_path=design_fasta_path,
            design_is_cyclic=design_is_cyclic,
            design_group_id=design_group_id,
            ligand_smiles=ligand_smiles,
            ligand_smiles_path=ligand_smiles_path,
            ligand_ccds=ligand_ccds,
            ligand_ccd_path=ligand_ccd_path,
        )
    else:
        if not design_spec and not design_spec_path:
            raise ValueError(
                "No explicit query inputs provided. Supply target/design/ligand inputs "
                "or provide `design_spec`/`design_spec_path`."
            )
        query = None
        query_metadata = {"query_chain_ids": [], "target_source": "none", "design_source": "none", "num_ligands": 0}

    result = run_boltzgen_design(
        query=query,
        design_spec=design_spec,
        design_spec_path=design_spec_path,
        n_structures=n_structures,
        out_dir=out_dir,
        out_summary_json=out_summary_json,
        output_prefix=output_prefix,
        session=session,
    )
    result["query_metadata"] = query_metadata
    return result


def _parse_list_arg(values: Optional[Iterable[str]]) -> List[str]:
    return [v for v in (values or []) if v and v.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BoltzGen design with OpenProtein API.")
    p.add_argument("--target-structure-path", type=Path, default=None)
    p.add_argument("--target-pdb-id", default=None)
    p.add_argument("--target-chain-id", default="A")
    p.add_argument("--target-sequence", default=None)
    p.add_argument("--target-fasta-path", type=Path, default=None)
    p.add_argument("--target-is-cyclic", action="store_true")
    p.add_argument("--design-structure-path", type=Path, default=None)
    p.add_argument("--design-pdb-id", default=None)
    p.add_argument("--design-source-chain-id", default=None)
    p.add_argument("--design-chain-id", default="B")
    p.add_argument("--design-sequence", default=None)
    p.add_argument("--design-fasta-path", type=Path, default=None)
    p.add_argument("--design-is-cyclic", action="store_true")
    p.add_argument("--design-group-id", type=int, default=1)
    p.add_argument("--ligand-smiles", action="append", default=[])
    p.add_argument("--ligand-smiles-path", type=Path, default=None)
    p.add_argument("--ligand-ccd", action="append", default=[])
    p.add_argument("--ligand-ccd-path", type=Path, default=None)
    p.add_argument("--design-spec-path", type=Path, default=None)
    p.add_argument("--n-structures", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/boltzgen"))
    p.add_argument("--out-summary", type=Path, default=Path("outputs/boltzgen/summary.json"))
    p.add_argument("--output-prefix", default="boltzgen_design")
    p.add_argument("--run-proteinmpnn", action="store_true")
    p.add_argument("--proteinmpnn-samples", type=int, default=8)
    p.add_argument("--proteinmpnn-out-csv", type=Path, default=Path("outputs/boltzgen/proteinmpnn_sequences.csv"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = design_with_boltzgen(
        target_chain_id=args.target_chain_id,
        target_structure_path=args.target_structure_path,
        target_pdb_id=args.target_pdb_id,
        target_sequence=args.target_sequence,
        target_fasta_path=args.target_fasta_path,
        target_is_cyclic=args.target_is_cyclic,
        design_chain_id=args.design_chain_id,
        design_structure_path=args.design_structure_path,
        design_pdb_id=args.design_pdb_id,
        design_source_chain_id=args.design_source_chain_id,
        design_sequence=args.design_sequence,
        design_fasta_path=args.design_fasta_path,
        design_is_cyclic=args.design_is_cyclic,
        design_group_id=args.design_group_id,
        ligand_smiles=_parse_list_arg(args.ligand_smiles),
        ligand_smiles_path=args.ligand_smiles_path,
        ligand_ccds=_parse_list_arg(args.ligand_ccd),
        ligand_ccd_path=args.ligand_ccd_path,
        design_spec_path=args.design_spec_path,
        n_structures=args.n_structures,
        out_dir=args.out_dir,
        out_summary_json=args.out_summary,
        output_prefix=args.output_prefix,
    )

    if args.run_proteinmpnn and result.get("structures") is not None:
        records = run_proteinmpnn_from_structures(
            generated_structures=result["structures"],
            target_chain_id=args.target_chain_id,
            design_chain_id=args.design_chain_id,
            num_samples=args.proteinmpnn_samples,
            out_csv=args.proteinmpnn_out_csv,
        )
        result["proteinmpnn_records"] = records[:5]
        result["proteinmpnn_n_records"] = len(records)

    printable = {k: v for k, v in result.items() if k != "structures" and k != "job"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
