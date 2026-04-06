"""OpenProtein helpers with lazy imports.

This package init stays lightweight so callers can import narrow helpers
without pulling in optional heavy dependencies such as PyMOL.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict


_EXPORTS: Dict[str, str] = {
    "predict_boltz2": "abcode.tools.openprotein.predict_boltz2_structure_openprotein",
    "build_boltzgen_query": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "boltzgen_yaml_to_design_kwargs": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "validate_design_with_boltzgen_kwargs": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "assert_valid_design_with_boltzgen_kwargs": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "design_with_boltzgen_yaml": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "evaluate_boltz2_refold_metrics": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "filter_and_select_designs": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "run_boltzgen_design": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "run_proteinmpnn_postdesign_pipeline": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "run_proteinmpnn_from_structures": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "design_with_boltzgen": "abcode.tools.openprotein.design_boltzgen_openprotein",
    "create_openprotein_msa": "abcode.tools.openprotein.align_msa_openprotein",
    "save_openprotein_msa": "abcode.tools.openprotein.align_msa_openprotein",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """
    Lazily resolve exported helpers on first access.

    Args:
        name: Exported symbol requested from this package.

    Returns:
        The resolved function or object from the underlying module.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
