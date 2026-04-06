from pymol import cmd
from pathlib import Path
from project_config.variables import mapping_rev, aa_polarity_mapping

def _resi_sort_key(resi_value):
    text = str(resi_value).strip()
    sign = -1 if text.startswith("-") else 1
    core = text[1:] if sign == -1 else text
    digits = ""
    suffix = ""
    for ch in core:
        if ch.isdigit() and not suffix:
            digits += ch
        else:
            suffix += ch
    if digits:
        return (sign * int(digits), suffix)
    return (10**9, text)

def _pdb_chain_ids(pdb_fpath):
    """Return chain IDs present in a structure file via PyMOL."""
    cmd.reinitialize()
    cmd.load(pdb_fpath, "obj")
    chains = [c.strip() for c in cmd.get_chains("obj") if str(c).strip()]
    return chains

# Section: format conversion ----------------------------------------------------
def convert_cif_to_pdb_pymol(cif_fpath, pdb_fpath):
    """
    Convert CIF fpath into PDB fpath.
    """
    cmd.reinitialize()
    cmd.load(cif_fpath)
    cmd.save(pdb_fpath)
    return pdb_fpath

# Section: residue selection utilities -----------------------------------------
def filter_residues(pdb_fpath, selection_condition="byres (chain A within 5 of chain B)", selection_name='selection'):
    """
    Filter a structure using a PyMOL selection and overwrite output PDB.

    Args:
        pdb_fpath: Input/output PDB path.
        selection_condition: PyMOL selection expression.
        selection_name: Temporary selection object name.

    Returns:
        None. Writes filtered structure to `pdb_fpath`.
    """
    cmd.reinitialize()
    cmd.load(pdb_fpath, "obj")
    cmd.select(selection_name, selection_condition)
    # Keep raw residue ids to avoid insertion-code mismatches (e.g. 44A vs 44).
    res_filt = sorted(
        {(str(a.resn).upper(), str(a.resi).strip()) for a in cmd.get_model(f"{selection_name} and name CA").atom},
        key=lambda x: _resi_sort_key(x[1]),
    )
    return res_filt

# Section: structure alignment utilities -----------------------------------------
def align_structures(pdb_fpath_target, pdb_fpath_ref):
    """
    Align target structure to reference using PyMOL `align`.

    Args:
        pdb_fpath_target: Target PDB path (moved/aligned in output).
        pdb_fpath_ref: Reference PDB path.

    Returns:
        None. Writes aligned target structure back to `pdb_fpath_target`.
    """
    cmd.reinitialize()
    cmd.load(pdb_fpath_target, "target")
    cmd.load(pdb_fpath_ref, "ref")
    cmd.align('target', 'ref')
    cmd.save(pdb_fpath_target, "target")
    print(f'Aligned "{pdb_fpath_target}" to "{pdb_fpath_ref}".')

# Section: default visualization ------------------------------------------------
def visualize_structures(pdb_list, show_res_near_ligand=6, protein_chain_id='A', ligand_chain_id='B'):
    """
    Visualize structures in pdb_list using NGLview
    """
    import nglview as nv
    view = nv.NGLWidget()
    for i, pdb in enumerate(pdb_list):
        print(i, pdb)
        # base context
        view.add_component(pdb, default_representation=False)
        view[i].clear_representations()
        view[i].add_cartoon(selection=f":{protein_chain_id}", colorScheme='residueindex')
        view[i].add_ball_and_stick(selection=':B', colorValue="orange")
        view[i].add_ball_and_stick(selection=':C', colorValue="orange")
        view[i].add_ball_and_stick(selection=f':{ligand_chain_id}')

        # color nearby protein residues close to ligand
        if show_res_near_ligand is not None:
            res_near_ligand = filter_residues(pdb, f"byres (chain {protein_chain_id} within {show_res_near_ligand} of chain {ligand_chain_id})", f'near_{protein_chain_id}_{ligand_chain_id}')
            for aa, res_num in res_near_ligand:
                aa_one = mapping_rev.get(str(aa).upper())
                if not aa_one:
                    continue
                polarity_grp = aa_polarity_mapping.get(aa_one)
                if not polarity_grp:
                    continue
                color = {"np": "lime", "p~": "cyan", "p+": "blue", "p-": "red"}[polarity_grp]
                view[i].add_licorice(
                    selection=f":{protein_chain_id} and {res_num} and {aa}",
                    colorValue=color,
                )
    view.center()
    return view


# Section: overlay visualization ------------------------------------------------
def visualize_overlay_structures(
    pdb_list,
    *,
    protein_chain_id='A',
    ligand_chain_id='B',
    protein_colors=None,
    protein_opacities=None,
    protein_radius_scales=None,
    ligand_color="grey",
    show_target_representation=True,
    target_representation="ball+stick",
    show_res_near_target=True,
    near_distance_angstrom=6,
):
    """
    Visualize one or more PDBs with solid-color protein cartoons.
    Useful for overlaying generated vs predicted structures with explicit colors.
    """
    import nglview as nv

    color_list = list(protein_colors or [])
    opacity_list = list(protein_opacities or [])
    radius_scale_list = list(protein_radius_scales or [])
    default_palette = ["#00bfff", "#ff6347", "#32cd32", "#ff00ff"]

    named_to_hex = {
        "deepskyblue": "#00bfff",
        "magenta": "#ff00ff",
        "tomato": "#ff6347",
        "limegreen": "#32cd32",
        "gold": "#ffd700",
        "orange": "#ffa500",
        "lime": "#00ff00",
        "cyan": "#00ffff",
        "blue": "#0000ff",
        "red": "#ff0000",
        "green": "#008000",
    }

    def _as_hex(color_value, fallback):
        text = str(color_value or "").strip()
        if not text:
            return fallback
        if text.startswith("#"):
            return text
        return named_to_hex.get(text.lower(), fallback)

    view = nv.NGLWidget()
    for i, pdb in enumerate(pdb_list):
        view.add_component(pdb, default_representation=True)
        view[i].clear_representations()
        chains = _pdb_chain_ids(pdb)
        protein_chain_for_comp = protein_chain_id if protein_chain_id in chains else (chains[0] if chains else None)
        remaining_chains = [c for c in chains if c != protein_chain_for_comp]
        target_chain_for_comp = ligand_chain_id if ligand_chain_id in chains else (remaining_chains[0] if remaining_chains else None)
        fallback_color = default_palette[i % len(default_palette)]
        color = _as_hex(color_list[i] if i < len(color_list) else fallback_color, fallback_color)
        opacity = float(opacity_list[i] if i < len(opacity_list) else (0.35 if i == 1 else 0.95))
        radius_scale = float(radius_scale_list[i] if i < len(radius_scale_list) else (0.85 if i == 1 else 1.0))
        print(i, pdb, protein_chain_for_comp, color)
        if protein_chain_for_comp:
            view[i].add_cartoon(
                selection=f":{protein_chain_for_comp} and protein",
                color=color,
                opacity=opacity,
                radiusScale=radius_scale,
            )
        if show_target_representation and target_chain_for_comp:
            view[i].add_representation(
                target_representation,
                selection=f":{target_chain_for_comp}",
                colorValue=ligand_color,
            )
        # Optionally highlight protein residues near target chain
        # (target can be ligand, peptide, or protein; selection is chain-based).
        if show_res_near_target and protein_chain_for_comp and target_chain_for_comp:
            try:
                res_near_target = filter_residues(
                    pdb,
                    f"byres (chain {protein_chain_for_comp} within {near_distance_angstrom} of chain {target_chain_for_comp})",
                    f"near_{protein_chain_for_comp}_{target_chain_for_comp}_{i}",
                )
                for aa, res_num in res_near_target:
                    aa_one = mapping_rev.get(str(aa).upper())
                    if not aa_one:
                        continue
                    polarity_grp = aa_polarity_mapping.get(aa_one)
                    if not polarity_grp:
                        continue
                    near_color = {"np": "lime", "p~": "cyan", "p+": "blue", "p-": "red"}[polarity_grp]
                    view[i].add_licorice(
                        selection=f":{protein_chain_for_comp} and {res_num} and {aa}",
                        colorValue=near_color
                    )
            except Exception:
                # Keep visualization robust even when residue extraction fails.
                pass
    view.center()
    return view


def run_tmalign(pdb_a, pdb_b, tmalign_exe='/Users/charmainechia/Documents/projects/TMalign-20180426/TMalign'):
    """
    Run TMalign on two PDB files.
    Args:
        pdb_a: First PDB filepath
        pdb_b: Second PDB filepath
        tmalign_exe: Path to TMalign executable (e.g., "./TMalign" or "/path/to/TMalign")
    Returns:
        subprocess.CompletedProcess with stdout/stderr as strings.
    """
    import subprocess
    tmalign_exe = str(Path(tmalign_exe))
    pdb_a = str(Path(pdb_a))
    pdb_b = str(Path(pdb_b))
    cmd = [tmalign_exe, pdb_a, pdb_b]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = res.stdout
    # parse TM score
    tmscore = float(stdout[stdout.find('TM-score= ')+10:stdout.find(' (if normalized by length of Chain_1')])
    return tmscore
