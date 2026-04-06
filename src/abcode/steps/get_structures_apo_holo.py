from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path

    _repo_root = Path(__file__).resolve().parents[3]
    _src_root = _repo_root / "src"
    for _path in (str(_repo_root), str(_src_root)):
        if _path not in sys.path:
            sys.path.insert(0, _path)

from typing import Any, Dict
from pathlib import Path
from abcode.tools.openprotein import predict_boltz2


def run_structure_prediction(inputs: Dict[str, Any], out_cif_path: Path, out_summary_path: Path) -> Dict[str, Any]:
    """
    Select and run structure-prediction backend for apo/holo workflow step.

    Args:
        inputs: Structure step configuration dict (tool choice, sequences, ligands).
        out_cif_path: Destination CIF path for generated structure.
        out_summary_path: Destination JSON summary path.

    Returns:
        Backend status/summary dictionary.
    """
    tool = str(inputs.get("preferred_structure_tool", "")).strip().lower()

    if tool == "rcsb_search":
        return {
            "tool": tool,
            "implemented": False,
            "message": "RCSB PDB search path is not implemented yet.",
        }

    if tool == "boltz2_local":
        return {
            "tool": tool,
            "implemented": False,
            "message": "Local Boltz-2 path is not implemented yet.",
        }

    if tool == "boltz2_openprotein":
        sequences = list(inputs.get("sequences", []))
        ligands = list(inputs.get("ligands", []))
        if not sequences:
            raise ValueError("No protein sequences provided.")

        summary = predict_boltz2(
            sequences=sequences,
            smiles=ligands,
            ccds=None,
            predict_affinity=bool(inputs.get("predict_affinity", False)),
            wait=True,
            out_cif=out_cif_path,
            out_summary_json=out_summary_path,
        )
        summary["tool"] = tool
        summary["implemented"] = True
        return summary

    return {
        "tool": tool,
        "implemented": False,
        "message": f"Unknown tool option: {tool}",
    }


if __name__ == "__main__":
    from abcode.core.pipeline_utils import print_run_summary, resolve_repo_root
    from project_config.variables import address_dict, subfolders

    root_key = "examples"
    user_inputs = {
        "pdb_id": "",
        "struct_name": "example_structure",
        "sequences": [],
        "ligands": [],
        "preferred_structure_tool": "boltz2_openprotein",
        "predict_affinity": False,
    }

    repo_root = resolve_repo_root(__file__)
    data_root = (repo_root / address_dict[root_key]).resolve()
    pdb_dir = (data_root / subfolders["pdb"]).resolve()
    pdb_dir.mkdir(parents=True, exist_ok=True)

    out_cif = pdb_dir / f"{user_inputs['struct_name']}_boltz2.cif"
    out_summary = pdb_dir / f"{user_inputs['struct_name']}_boltz2_summary.json"

    result = run_structure_prediction(user_inputs, out_cif, out_summary)
    print_run_summary(
        {
            "root_key": root_key,
            "pdb_dir": pdb_dir,
            "out_cif": out_cif,
            "out_summary": out_summary,
            "result": result,
        }
    )
