from __future__ import annotations

from pathlib import Path
from typing import Dict, MutableMapping, Sequence

from project_config.variables import address_dict, subfolders


def resolve_input_path(data_root: Path, path_value: str) -> Path:
    """Resolve an input path relative to a selected data root."""
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (data_root / p).resolve()


def _read_nonempty_lines(path: Path) -> Sequence[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_project_root() -> Path:
    """
    Resolve the repository root robustly from different working directories.

    Returns:
        Absolute path to the project root directory.
    """
    cwd = Path.cwd().resolve()

    # Walk up parents to find a likely repository root marker set.
    for candidate in [cwd, *cwd.parents]:
        has_git = (candidate / ".git").exists()
        has_project_config = (candidate / "project_config").exists()
        has_src = (candidate / "src").exists()
        if has_git or (has_project_config and has_src):
            return candidate

    # Fallback to previous behavior if no marker is found.
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def setup_data_root(
    root_key: str,
    required_subfolders: Sequence[str],
    project_root: Path | None = None,
) -> tuple[Path, Dict[str, Path]]:
    """
    Resolve and create a selected data root plus a standard set of subfolders.

    Args:
        root_key: Key in `project_config.variables.address_dict`.
        required_subfolders: Iterable of keys from `subfolders` to resolve/create.
        project_root: Optional explicit project-root override.

    Returns:
        Tuple `(data_root, resolved_subfolder_paths)`.
    """
    if root_key not in address_dict:
        raise KeyError(f"Unknown root_key: {root_key}")
    base = project_root or resolve_project_root()
    data_root = (base / address_dict[root_key]).resolve()
    resolved: Dict[str, Path] = {}
    for key in required_subfolders:
        folder = data_root / subfolders[key]
        folder.mkdir(parents=True, exist_ok=True)
        resolved[key] = folder
    return data_root, resolved


def setup_step_data_root(
    root_key: str,
    default_required_subfolders: Sequence[str],
    required_subfolders: Sequence[str] | None = None,
    project_root: Path | None = None,
) -> tuple[Path, Dict[str, Path]]:
    """
    Resolve a step data root using a step's default subfolder set unless overridden.
    """
    return setup_data_root(
        root_key=root_key,
        required_subfolders=required_subfolders or default_required_subfolders,
        project_root=project_root,
    )


def get_step_processed_dir(resolved_dirs: Dict[str, Path], step_output_subdir: str) -> Path:
    """
    Create and return a step-specific processed output directory.
    """
    step_dir = (resolved_dirs["processed"] / str(step_output_subdir).strip()).resolve()
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def join_data_path(data_root: Path, subdir: str, data_subfolder: str, filename: str) -> Path:
    """
    Join a selected data root with a standard subdir, optional nested subfolder, and filename.

    Args:
        data_root: Resolved data root.
        subdir: Relative subdirectory path under the data root.
        data_subfolder: Optional nested project-specific subfolder.
        filename: Filename or trailing relative path.

    Returns:
        Resolved filesystem path.
    """
    sub = str(subdir).strip().strip("/")
    ds = str(data_subfolder or "").strip().strip("/")
    name = str(filename).strip().lstrip("/")
    p = data_root
    if sub:
        p = p / sub
    if ds:
        p = p / ds
    return p / name


def apply_optional_text_inputs(
    user_inputs: MutableMapping[str, object],
    input_paths: Dict[str, str],
    data_root: Path,
) -> MutableMapping[str, object]:
    """
    Populate user_inputs from optional text files when provided.
    - seed_sequences_file: one sequence name per line
    - constraints_file: one constraint per line
    """
    seed_file_val = str(input_paths.get("seed_sequences_file", "")).strip()
    if seed_file_val:
        seed_file = resolve_input_path(data_root, seed_file_val)
        if seed_file.exists():
            seed_lines = _read_nonempty_lines(seed_file)
            if seed_lines:
                user_inputs["seed_sequences"] = list(seed_lines)

    constraints_file_val = str(input_paths.get("constraints_file", "")).strip()
    if constraints_file_val:
        constraints_file = resolve_input_path(data_root, constraints_file_val)
        if constraints_file.exists():
            constraint_lines = _read_nonempty_lines(constraints_file)
            if constraint_lines:
                user_inputs["constraints"] = list(constraint_lines)

    return user_inputs
