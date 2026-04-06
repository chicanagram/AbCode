#!/usr/bin/env python3
"""Prepare mutagenesis datasets and build single-mutation segmentation indices.

This script implements the first part of a split-preparation workflow:
1) Load and validate CSV columns.
2) Build one-hot mutated-position columns.
3) Sort and print dataset-level stats.
4) Segment single-mutation mutants by position groups.
5) Optionally sub-segment the smallest single-mutation segment.
6) Print final list-of-lists index groups.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


REQUIRED_COLUMNS = {"num_mutations", "positions", "mutations"}


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process mutagenesis CSV and prepare single-mutation position segments."
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Directory containing the CSV file.",
    )
    parser.add_argument(
        "--csv-file",
        default=None,
        help="CSV filename.",
    )
    parser.add_argument(
        "--mutation-separator",
        default=None,
        help="Separator for mutation tokens in the 'mutations' column (default=':').",
    )
    parser.add_argument(
        "--num-mutation-segments-singlemut",
        type=int,
        default=None,
        help="Number of primary segments to create at the single-mutation level.",
    )
    parser.add_argument(
        "--min-layer-size-for-multimut-segmentation",
        type=int,
        default=None,
        help=(
            "Minimum layer-block size used to merge adjacent num_mutations layers."
        ),
    )
    parser.add_argument(
        "--max-layer-size-for-multimut-segmentation",
        type=int,
        default=None,
        help=(
            "Target maximum segment size used for within-layer segmentation. "
            "If unset/None, no within-layer segmentation is performed."
        ),
    )
    parser.add_argument(
        "--smallest-single-mutant-size",
        type=int,
        default=None,
        help="Target smallest dataset size at the single-mutation level.",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=None,
        help="K value for fold column naming (e.g. 5 -> fold_random_5).",
    )
    parser.add_argument(
        "--include-mutation-onehot-for-clustering",
        type=parse_bool,
        default=None,
        help=(
            "If True, multi-mutant k-means uses concatenated "
            "[position_one_hot || mutation_one_hot] features."
        ),
    )
    return parser.parse_args()


def parse_positions(positions_value: object) -> List[int]:
    """Parse 'positions' cell into sorted unique integer positions."""
    if pd.isna(positions_value):
        return []
    raw = str(positions_value).strip()
    if not raw:
        return []

    # Positions in provided datasets are usually comma-separated numeric strings.
    tokens = [tok.strip() for tok in raw.replace(";", ",").split(",") if tok.strip()]

    parsed: List[int] = []
    for tok in tokens:
        if tok.lstrip("-").isdigit():
            parsed.append(int(tok))
            continue

        digits = "".join(ch for ch in tok if ch.isdigit() or ch == "-")
        if digits and digits.lstrip("-").isdigit():
            parsed.append(int(digits))

    return sorted(set(parsed))


def parse_mutation_tokens(mutations_value: object, separator: str) -> List[str]:
    if pd.isna(mutations_value):
        return []
    raw = str(mutations_value).strip()
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(separator) if tok.strip()]


def load_and_validate_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. Found columns: {list(df.columns)}"
        )
    return df


def add_position_one_hot(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    parsed_positions = df["positions"].apply(parse_positions)
    all_positions = sorted({pos for row in parsed_positions for pos in row})

    one_hot = pd.DataFrame(0, index=df.index, columns=[f"pos_{p}" for p in all_positions])
    for idx, row_positions in parsed_positions.items():
        for pos in row_positions:
            one_hot.at[idx, f"pos_{pos}"] = 1

    out = df.copy()
    out = pd.concat([out, one_hot], axis=1)
    return out, all_positions


def build_mutation_one_hot(df: pd.DataFrame, mutation_separator: str) -> pd.DataFrame:
    mutation_tokens_per_row = df["mutations"].apply(
        lambda x: parse_mutation_tokens(x, mutation_separator)
    )
    unique_mutations = sorted({m for muts in mutation_tokens_per_row for m in muts})
    mut_one_hot = pd.DataFrame(
        0, index=df.index, columns=[f"mut_{m}" for m in unique_mutations], dtype="int8"
    )
    for idx, muts in mutation_tokens_per_row.items():
        for m in muts:
            mut_one_hot.at[idx, f"mut_{m}"] = 1
    return mut_one_hot


def print_overall_stats(df: pd.DataFrame, mutation_separator: str, all_positions: Sequence[int]) -> None:
    unique_mutants = df["mutations"].astype(str).nunique(dropna=True)

    mutation_token_set = set()
    for val in df["mutations"]:
        mutation_token_set.update(parse_mutation_tokens(val, mutation_separator))

    total = len(df)
    num_mut_dist = (
        df["num_mutations"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("num_mutations")
        .reset_index(name="count")
    )

    print("\n=== Dataset Stats ===")
    print(f"Total mutants: {total}")
    print(f"Unique mutants: {unique_mutants}")
    print(f"Unique individual mutation tokens: {len(mutation_token_set)}")
    print(f"Unique mutated positions: {len(all_positions)}")
    print("Percent of mutants by num_mutations:")
    for _, row in num_mut_dist.iterrows():
        pct = (row["count"] / total) * 100 if total else 0.0
        print(f"  num_mutations={row['num_mutations']}: {row['count']} ({pct:.2f}%)")


def assign_round_robin(items: Sequence[int], n_groups: int) -> Dict[int, int]:
    if n_groups <= 0:
        raise ValueError("n_groups must be > 0")
    return {item: i % n_groups for i, item in enumerate(items)}


def segment_single_mutants(
    df: pd.DataFrame,
    n_segments: int,
    target_smallest_size: int,
) -> Tuple[List[Tuple[str, List[int]]], pd.DataFrame]:
    one_mut_df = df[df["num_mutations"] == 1].copy()
    if one_mut_df.empty:
        raise ValueError("No single-mutation entries found (num_mutations == 1).")

    one_mut_df["parsed_positions"] = one_mut_df["positions"].apply(parse_positions)
    one_mut_df = one_mut_df[one_mut_df["parsed_positions"].map(len) > 0].copy()
    if one_mut_df.empty:
        raise ValueError("Single-mutation entries exist, but no parseable positions were found.")

    one_mut_df["single_position"] = one_mut_df["parsed_positions"].apply(lambda x: x[0])
    unique_single_positions = sorted(one_mut_df["single_position"].unique().tolist())
    pos_to_seg = assign_round_robin(unique_single_positions, n_segments)
    one_mut_df["segment_id"] = one_mut_df["single_position"].map(pos_to_seg)

    segment_groups: Dict[int, List[int]] = {
        seg_id: grp.index.tolist() for seg_id, grp in one_mut_df.groupby("segment_id")
    }

    print("\n=== Single-Mutation Segmentation (Primary) ===")
    print(f"Unique positions among single mutants: {len(unique_single_positions)}")
    print(f"Configured segments: {n_segments}")
    for seg_id in sorted(segment_groups):
        print(f"  segment_{seg_id}: {len(segment_groups[seg_id])} mutants")

    smallest_seg_id, smallest_indices = min(segment_groups.items(), key=lambda kv: len(kv[1]))
    smallest_size = len(smallest_indices)

    result_groups: List[Tuple[str, List[int]]] = []

    if abs(smallest_size - target_smallest_size) <= 10:
        print(
            "Smallest segment is within +/-10 of target "
            f"({smallest_size} vs target {target_smallest_size}); skipping sub-segmentation."
        )
        for seg_id, idxs in sorted(segment_groups.items(), key=lambda kv: len(kv[1])):
            result_groups.append((f"segment_{seg_id}", sorted(idxs)))
        return result_groups, one_mut_df

    print(
        "Smallest segment is not within +/-10 of target "
        f"({smallest_size} vs target {target_smallest_size})."
    )
    print("Sub-segmenting the smallest segment.")

    best_divider = find_best_divider(smallest_size, target_smallest_size)
    if best_divider <= 1:
        print(
            "Best whole-number divider is 1; cannot sub-segment further while meeting "
            "the requested rule."
        )
        for seg_id, idxs in sorted(segment_groups.items(), key=lambda kv: len(kv[1])):
            result_groups.append((f"segment_{seg_id}", sorted(idxs)))
        return result_groups, one_mut_df

    print(f"Chosen divider for smallest segment: {best_divider}")

    smallest_df = one_mut_df.loc[smallest_indices].copy()
    smallest_positions = sorted(smallest_df["single_position"].unique().tolist())
    pos_to_subseg = assign_round_robin(smallest_positions, best_divider)
    smallest_df["subsegment_id"] = smallest_df["single_position"].map(pos_to_subseg)

    subgroups: List[Tuple[str, List[int]]] = []
    for sub_id, grp in smallest_df.groupby("subsegment_id"):
        subgroups.append((f"segment_{smallest_seg_id}_sub_{sub_id}", sorted(grp.index.tolist())))

    subgroups.sort(key=lambda kv: len(kv[1]))

    print("Sub-segment sizes for smallest segment:")
    for label, idxs in subgroups:
        print(f"  {label}: {len(idxs)} mutants")

    result_groups.extend(subgroups)

    for seg_id, idxs in sorted(segment_groups.items(), key=lambda kv: len(kv[1])):
        if seg_id == smallest_seg_id:
            continue
        result_groups.append((f"segment_{seg_id}", sorted(idxs)))

    return result_groups, one_mut_df


def find_best_divider(group_size: int, target_size: int) -> int:
    if group_size <= 1:
        return 1
    if target_size <= 0:
        return 1

    best = 1
    best_err = math.inf
    for divider in range(1, group_size + 1):
        projected = group_size / divider
        err = abs(projected - target_size)
        if err < best_err:
            best = divider
            best_err = err
    return best


def sort_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(by=["num_mutations", "positions", "mutations"], kind="stable")
        .reset_index(drop=True)
    )


def rename_groups_sequentially(
    index_groups: Sequence[Tuple[str, Sequence[int]]],
) -> List[Tuple[str, List[int]]]:
    renamed: List[Tuple[str, List[int]]] = []
    for i, (_, idxs) in enumerate(index_groups):
        renamed.append((f"segment_{i}", sorted(list(idxs))))
    return renamed


def print_index_group_list(
    index_groups: Sequence[Tuple[str, Sequence[int]]],
    df: pd.DataFrame,
) -> None:
    print("\n=== Final Index Groups (All Layers) ===")
    print("Ordered groups and index lists:")
    cumulative_n = 0
    for i, (label, idxs) in enumerate(index_groups, start=1):
        idxs_list = list(idxs)
        positions = sorted(
            {
                p
                for pos_text in df.loc[idxs_list, "positions"]
                for p in parse_positions(pos_text)
            }
        )
        cumulative_n += len(idxs_list)
        print(f"  {i}. {label} (n={len(idxs_list)}, cumulative_n={cumulative_n})")
        print(f"     positions_count={len(positions)} positions={positions}")
        print(f"     indices={idxs_list}")


def print_index_group_summary(index_groups: Sequence[Tuple[str, Sequence[int]]]) -> None:
    print("\n=== Final Index Groups (All Layers) ===")
    print("Ordered groups and sizes:")
    cumulative_n = 0
    for i, (label, idxs) in enumerate(index_groups, start=1):
        n = len(list(idxs))
        cumulative_n += n
        print(f"  {i}. {label}: n={n}, cumulative_n={cumulative_n}")


def print_fold_label_counts(df: pd.DataFrame, k_folds: int) -> None:
    fold_cols = [f"fold_random_{k_folds}", f"fold_mutres-modulo_{k_folds}"]
    print("\n=== Fold Label Counts (Overall Dataframe) ===")
    for col in fold_cols:
        print(f"{col}:")
        counts = df[col].value_counts(dropna=False).sort_index()
        for label, n in counts.items():
            label_str = "NA" if pd.isna(label) else int(label)
            print(f"  label={label_str}: {int(n)}")


def build_segment_index_series(
    *,
    n_rows: int,
    index_groups: Sequence[Tuple[str, Sequence[int]]],
) -> pd.Series:
    out = pd.Series([pd.NA] * n_rows, dtype="Int64")
    for segment_idx, (_, idxs) in enumerate(index_groups):
        out.iloc[list(idxs)] = segment_idx
    return out


def fit_kmeans_with_rebalance(features: np.ndarray, n_clusters: int) -> np.ndarray:
    n_samples = len(features)
    if n_samples == 0:
        return np.array([], dtype=int)
    if n_clusters <= 1 or n_samples == 1:
        return np.zeros(n_samples, dtype=int)

    model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = model.fit_predict(features)
    labels = rebalance_kmeans_labels(
        labels=labels,
        features=features,
        centroids=model.cluster_centers_,
        n_clusters=n_clusters,
    )
    return labels


def build_clustering_features(
    *,
    df: pd.DataFrame,
    row_indices: Sequence[int],
    position_columns: Sequence[str],
    include_mutation_onehot_for_clustering: bool,
    mutation_one_hot: pd.DataFrame | None,
) -> np.ndarray:
    x_pos = df.loc[list(row_indices), list(position_columns)].to_numpy()
    if not include_mutation_onehot_for_clustering:
        return x_pos
    if mutation_one_hot is None:
        raise ValueError("mutation_one_hot is required when include_mutation_onehot_for_clustering=True")
    x_mut = mutation_one_hot.loc[list(row_indices)].to_numpy()
    return np.concatenate([x_pos, x_mut], axis=1)


def populate_fold_labels(
    df: pd.DataFrame,
    k_folds: int,
    position_columns: Sequence[str],
    include_mutation_onehot_for_clustering: bool,
    segment_col: str = "segment_index_0",
    mutation_one_hot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    pos_cols = list(position_columns)
    fold_random_col = f"fold_random_{k_folds}"
    fold_mutres_mod_col = f"fold_mutres-modulo_{k_folds}"

    if segment_col not in out.columns:
        raise KeyError(f"segment column '{segment_col}' not found in dataframe.")
    assigned_df = out[out[segment_col].notna()].copy()
    if assigned_df.empty:
        return out

    # Single-mutants: keep existing segment-wise assignment behavior.
    single_df = assigned_df[assigned_df["num_mutations"] == 1]
    for segment_idx in sorted(single_df[segment_col].dropna().unique().tolist()):
        seg_rows = single_df[single_df[segment_col] == segment_idx]
        seg_indices = seg_rows.index.tolist()

        for i, row_idx in enumerate(seg_indices):
            out.at[row_idx, fold_random_col] = i % k_folds

        unique_positions_in_order: List[int] = []
        for row_idx in seg_indices:
            for pos in parse_positions(out.at[row_idx, "positions"]):
                if pos not in unique_positions_in_order:
                    unique_positions_in_order.append(pos)
        if not unique_positions_in_order:
            continue
        pos_to_fold = {pos: i % k_folds for i, pos in enumerate(unique_positions_in_order)}

        for row_idx in seg_indices:
            row_positions = parse_positions(out.at[row_idx, "positions"])
            if row_positions:
                out.at[row_idx, fold_mutres_mod_col] = pos_to_fold[row_positions[0]]

    # Multi-mutants: assign folds by num_mutations layer.
    multi_df = assigned_df[assigned_df["num_mutations"] > 1]
    layer_values = sorted(multi_df["num_mutations"].dropna().unique().tolist())
    for layer in layer_values:
        layer_rows = multi_df[multi_df["num_mutations"] == layer]
        layer_indices = layer_rows.index.tolist()

        # fold_random_k: sequential assignment down mutants in this layer.
        for i, row_idx in enumerate(layer_indices):
            out.at[row_idx, fold_random_col] = i % k_folds

        # fold_mutres-modulo_k: within each segment in this layer, cluster into k folds.
        layer_segments = sorted(layer_rows[segment_col].dropna().unique().tolist())
        for segment_idx in layer_segments:
            seg_rows = layer_rows[layer_rows[segment_col] == segment_idx]
            seg_indices = seg_rows.index.tolist()
            if not seg_indices:
                continue

            x = build_clustering_features(
                df=out,
                row_indices=seg_indices,
                position_columns=pos_cols,
                include_mutation_onehot_for_clustering=include_mutation_onehot_for_clustering,
                mutation_one_hot=mutation_one_hot,
            )

            n_clusters = min(k_folds, len(seg_indices))
            seg_fold_labels = fit_kmeans_with_rebalance(x, n_clusters)
            for row_idx, fold_label in zip(seg_indices, seg_fold_labels):
                out.at[row_idx, fold_mutres_mod_col] = int(fold_label)

    return out


def segment_multi_mutants_by_layer(
    df: pd.DataFrame,
    min_layer_size_for_multimut_segmentation: int,
    max_layer_size_for_multimut_segmentation: int | None,
    position_columns: Sequence[str],
    include_mutation_onehot_for_clustering: bool,
    mutation_one_hot: pd.DataFrame | None = None,
) -> List[Tuple[str, List[int]]]:
    if min_layer_size_for_multimut_segmentation <= 0:
        raise ValueError("--min-layer-size-for-multimut-segmentation must be > 0")
    if max_layer_size_for_multimut_segmentation is not None and max_layer_size_for_multimut_segmentation <= 0:
        raise ValueError("--max-layer-size-for-multimut-segmentation must be > 0 when provided")

    pos_cols = list(position_columns)
    result_groups: List[Tuple[str, List[int]]] = []
    layers = sorted(
        int(x) for x in df["num_mutations"].dropna().unique().tolist() if int(x) > 1
    )
    layer_counts = {layer: int(np.sum(df["num_mutations"] == layer)) for layer in layers}

    # Build contiguous layer blocks. If a layer is below threshold, merge upwards
    # while the running total stays below or equal to threshold. Stop right before
    # the threshold would be exceeded by the next layer.
    layer_blocks: List[List[int]] = []
    i = 0
    while i < len(layers):
        start_layer = layers[i]
        running_layers = [start_layer]
        running_total = layer_counts[start_layer]

        if running_total < min_layer_size_for_multimut_segmentation:
            j = i + 1
            while j < len(layers):
                next_layer = layers[j]
                next_count = layer_counts[next_layer]
                if running_total + next_count > min_layer_size_for_multimut_segmentation:
                    break
                running_layers.append(next_layer)
                running_total += next_count
                j += 1
            i = j
        else:
            i += 1

        layer_blocks.append(running_layers)

    print("\n=== Multi-Mutation Segmentation (KMeans by num_mutations Layer) ===")
    print(
        "Minimum layer-block size for merging adjacent layers: "
        f"{min_layer_size_for_multimut_segmentation}"
    )
    print(
        "Maximum layer size for within-layer segmentation: "
        f"{max_layer_size_for_multimut_segmentation}"
    )
    print(
        "Clustering feature set: "
        + (
            "[position_one_hot || mutation_one_hot]"
            if include_mutation_onehot_for_clustering
            else "[position_one_hot]"
        )
    )

    print(f"Layer blocks for segmentation: {layer_blocks}")

    for block_layers in layer_blocks:
        layer_df = df[df["num_mutations"].isin(block_layers)].copy()
        if layer_df.empty:
            continue

        n_samples = len(layer_df)
        if max_layer_size_for_multimut_segmentation is None:
            n_clusters = 1
        else:
            approx = float(n_samples) / float(max_layer_size_for_multimut_segmentation)
            k_floor = max(1, int(np.floor(approx)))
            k_ceil = max(1, int(np.ceil(approx)))
            k_round = max(1, int(round(approx)))
            candidates = sorted({1, k_floor, k_ceil, k_round})
            n_clusters = min(
                n_samples,
                min(
                    candidates,
                    key=lambda k: (
                        abs((float(n_samples) / float(k)) - float(max_layer_size_for_multimut_segmentation)),
                        abs(float(k) - approx),
                        k,
                    ),
                ),
            )
        equal_size = n_samples / n_clusters
        base = n_samples // n_clusters
        remainder = n_samples % n_clusters
        target_counts = {c: base + (1 if c < remainder else 0) for c in range(n_clusters)}
        if n_clusters == 1:
            labels = np.zeros(n_samples, dtype=int)
            pre_counts = {0: int(n_samples)}
            post_counts = {0: int(n_samples)}
        else:
            x = build_clustering_features(
                df=df,
                row_indices=layer_df.index.tolist(),
                position_columns=pos_cols,
                include_mutation_onehot_for_clustering=include_mutation_onehot_for_clustering,
                mutation_one_hot=mutation_one_hot,
            )
            model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            labels = model.fit_predict(x)
            pre_counts = {c: int(np.sum(labels == c)) for c in range(n_clusters)}
            labels = rebalance_kmeans_labels(
                labels=labels,
                features=x,
                centroids=model.cluster_centers_,
                n_clusters=n_clusters,
            )
            post_counts = {c: int(np.sum(labels == c)) for c in range(n_clusters)}
        layer_df["kmeans_cluster"] = labels

        if len(block_layers) == 1:
            block_label = f"{block_layers[0]}"
        else:
            block_label = f"{block_layers[0]}-{block_layers[-1]}"
        print(f"num_mutations_block={block_label}: n_samples={n_samples}, n_clusters={n_clusters}")
        print(
            f"  equal_size={equal_size:.2f}, target_counts={target_counts}, "
            f"pre_counts={pre_counts}, post_counts={post_counts}"
        )
        layer_groups: List[Tuple[str, List[int]]] = []
        for cluster_label, grp in layer_df.groupby("kmeans_cluster"):
            idxs = sorted(grp.index.tolist())
            print(f"  cluster_{int(cluster_label)}: {len(idxs)} mutants")
            layer_groups.append((f"num_mut_{block_label}_cluster_{int(cluster_label)}", idxs))

        layer_groups.sort(key=lambda kv: len(kv[1]))
        result_groups.extend(layer_groups)

    return result_groups


def rebalance_kmeans_labels(
    labels: np.ndarray,
    features: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Rebalance k-means clusters towards equal-sized quotas using centroid-guided transfers."""
    updated = labels.copy()
    n_samples = len(updated)
    if n_clusters <= 1 or n_samples == 0:
        return updated

    # Precompute distances from every sample to every centroid once.
    # Shape: (n_samples, n_clusters)
    centroid_distances = np.linalg.norm(
        features[:, None, :] - centroids[None, :, :],
        axis=2,
    )

    base = n_samples // n_clusters
    remainder = n_samples % n_clusters
    target_counts = {c: base + (1 if c < remainder else 0) for c in range(n_clusters)}

    def cluster_counts(current: np.ndarray) -> Dict[int, int]:
        return {c: int(np.sum(current == c)) for c in range(n_clusters)}

    counts = cluster_counts(updated)

    recipients = [c for c in range(n_clusters) if counts[c] < target_counts[c]]
    recipient_idx = 0

    while recipients:
        donors = [c for c in range(n_clusters) if counts[c] > target_counts[c]]
        if not donors:
            break

        donor = max(donors, key=lambda c: counts[c] - target_counts[c])
        if counts[donor] <= target_counts[donor]:
            break

        # Find next undersized recipient in sequential order.
        checked = 0
        recipient = None
        while checked < len(recipients):
            cand = recipients[recipient_idx % len(recipients)]
            if counts[cand] < target_counts[cand]:
                recipient = cand
                break
            recipient_idx += 1
            checked += 1
        if recipient is None:
            break

        donor_indices = np.where(updated == donor)[0]
        if donor_indices.size == 0:
            break

        # Transfer the donor sample closest to recipient centroid
        # using the precomputed centroid-distance matrix.
        donor_to_recipient_dist = centroid_distances[donor_indices, recipient]
        move_idx = int(donor_indices[int(np.argmin(donor_to_recipient_dist))])
        updated[move_idx] = recipient

        counts[donor] -= 1
        counts[recipient] += 1

        # Advance recipient pointer once after a successful transfer.
        recipient_idx += 1
        recipients = [c for c in range(n_clusters) if counts[c] < target_counts[c]]

    return updated


def main(defaults: Dict[str, object] | None = None) -> None:
    defaults = defaults or {}
    args = parse_args()
    if args.csv_dir is None:
        args.csv_dir = defaults.get("csv_dir")
    if args.csv_file is None:
        args.csv_file = defaults.get("csv_file")
    if args.mutation_separator is None:
        args.mutation_separator = defaults.get("mutation_separator", ":")
    if args.num_mutation_segments_singlemut is None:
        args.num_mutation_segments_singlemut = defaults.get("num_mutation_segments_singlemut", 5)
    if args.min_layer_size_for_multimut_segmentation is None:
        args.min_layer_size_for_multimut_segmentation = defaults.get(
            "min_layer_size_for_multimut_segmentation", 1000
        )
    if args.max_layer_size_for_multimut_segmentation is None:
        args.max_layer_size_for_multimut_segmentation = defaults.get(
            "max_layer_size_for_multimut_segmentation", None
        )
    if args.smallest_single_mutant_size is None:
        args.smallest_single_mutant_size = defaults.get("smallest_single_mutant_size", 100)
    if args.k_folds is None:
        args.k_folds = defaults.get("k_folds", 5)
    if args.include_mutation_onehot_for_clustering is None:
        args.include_mutation_onehot_for_clustering = defaults.get(
            "include_mutation_onehot_for_clustering", True
        )

    if not args.csv_dir or not args.csv_file:
        raise ValueError(
            "CSV location is not set. Provide --csv-dir/--csv-file or edit defaults in __main__."
        )

    csv_path = Path(args.csv_dir).expanduser().resolve() / args.csv_file
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_path = run_segmentation_pipeline(
        csv_path=csv_path,
        mutation_separator=str(args.mutation_separator),
        num_mutation_segments_singlemut=int(args.num_mutation_segments_singlemut),
        min_layer_size_for_multimut_segmentation=int(args.min_layer_size_for_multimut_segmentation),
        max_layer_size_for_multimut_segmentation=(
            None
            if args.max_layer_size_for_multimut_segmentation in (None, "")
            else int(args.max_layer_size_for_multimut_segmentation)
        ),
        smallest_single_mutant_size=int(args.smallest_single_mutant_size),
        k_folds=int(args.k_folds),
        include_mutation_onehot_for_clustering=bool(args.include_mutation_onehot_for_clustering),
    )
    print(f"Saved processed dataframe CSV: {out_path}")


def run_segmentation_pipeline(
    *,
    csv_path: Path,
    mutation_separator: str = ":",
    num_mutation_segments_singlemut: int = 5,
    min_layer_size_for_multimut_segmentation: int = 1000,
    max_layer_size_for_multimut_segmentation: int | None | List[int | None] = None,
    smallest_single_mutant_size: int = 100,
    k_folds: int = 5,
    include_mutation_onehot_for_clustering: bool = True,
    output_csv_path: Path | None = None,
    verbose: bool = True,
    print_group_details: bool = True,
) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if k_folds <= 0:
        raise ValueError("k_folds must be > 0")

    if verbose:
        print(f"Loading CSV from: {csv_path}")
    df = load_and_validate_dataframe(csv_path)
    original_columns = list(df.columns)
    original_index_col_candidates = [c for c in original_columns if str(c).startswith("Unnamed:")]
    original_index_col = original_index_col_candidates[0] if original_index_col_candidates else None
    # Preserve original row order even after temporary sorting for segmentation.
    df["_original_row_order"] = np.arange(len(df), dtype=int)

    df, all_positions = add_position_one_hot(df)
    df = sort_dataframe(df)

    if verbose:
        print_overall_stats(df, mutation_separator, all_positions)

    single_groups, _ = segment_single_mutants(
        df=df,
        n_segments=num_mutation_segments_singlemut,
        target_smallest_size=smallest_single_mutant_size,
    )
    position_columns = [c for c in df.columns if c.startswith("pos_")]
    mutation_one_hot = (
        build_mutation_one_hot(df, mutation_separator)
        if include_mutation_onehot_for_clustering
        else None
    )

    if isinstance(max_layer_size_for_multimut_segmentation, list):
        max_layer_schemes = list(max_layer_size_for_multimut_segmentation)
    else:
        max_layer_schemes = [max_layer_size_for_multimut_segmentation]
    if not max_layer_schemes:
        max_layer_schemes = [None]

    segment_cols: List[str] = []
    segmentation_schemes: List[Tuple[str, int | None, Sequence[Tuple[str, Sequence[int]]]]] = []
    for i, max_layer_size in enumerate(max_layer_schemes):
        multi_groups = segment_multi_mutants_by_layer(
            df=df,
            min_layer_size_for_multimut_segmentation=min_layer_size_for_multimut_segmentation,
            max_layer_size_for_multimut_segmentation=max_layer_size,
            position_columns=position_columns,
            include_mutation_onehot_for_clustering=include_mutation_onehot_for_clustering,
            mutation_one_hot=mutation_one_hot,
        )
        index_groups = rename_groups_sequentially(single_groups + multi_groups)
        seg_col = f"segment_index_{i}"
        df[seg_col] = build_segment_index_series(n_rows=len(df), index_groups=index_groups)
        segment_cols.append(seg_col)
        segmentation_schemes.append((seg_col, max_layer_size, index_groups))

    if "segment_index_0" not in df.columns:
        raise ValueError("Expected segment_index_0 to be present after segmentation.")

    # Fold labels are generated from the first segmentation scheme.
    fold_random_col = f"fold_random_{k_folds}"
    fold_mutres_mod_col = f"fold_mutres-modulo_{k_folds}"
    df[fold_random_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df[fold_mutres_mod_col] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df = populate_fold_labels(
        df=df,
        k_folds=k_folds,
        segment_col="segment_index_0",
        position_columns=position_columns,
        include_mutation_onehot_for_clustering=include_mutation_onehot_for_clustering,
        mutation_one_hot=mutation_one_hot,
    )

    if verbose:
        for seg_col, max_layer_size, index_groups in segmentation_schemes:
            print(
                "\n=== Segmentation Scheme ===\n"
                f"column={seg_col} | max_layer_size_for_multimut_segmentation={max_layer_size}"
            )
            if print_group_details:
                print_index_group_list(index_groups, df)
            else:
                print_index_group_summary(index_groups)
        print("\n=== Appended Columns ===")
        seg_cols_fmt = ", ".join(segment_cols)
        print(
            "Added columns: "
            f"{seg_cols_fmt}, "
            f"{fold_random_col}, {fold_mutres_mod_col}"
        )
        print(
            "segment_index_<i> populated for each segmentation scheme; "
            "fold columns are populated using segment_index_0."
        )
        print_fold_label_counts(df, k_folds)

    # Restore original dataset order so fold labels align with precomputed encoding arrays.
    df = df.sort_values(by="_original_row_order", kind="stable").reset_index(drop=True)
    df = df.drop(columns=["_original_row_order"], errors="ignore")

    keep_cols = list(original_columns) + segment_cols + [fold_random_col, fold_mutres_mod_col]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df.loc[:, keep_cols].copy()

    # Keep legacy index column if present, but avoid saving as "Unnamed:*".
    if original_index_col and original_index_col in df.columns:
        df = df.rename(columns={original_index_col: "row_index"})

    out_path = output_csv_path or csv_path.with_name(f"{csv_path.stem}_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    return int(s)


def _parse_optional_int_or_list(value: Any) -> int | None | List[int | None]:
    if isinstance(value, list):
        return [_parse_optional_int(v) for v in value]
    return _parse_optional_int(value)


def run_data_segmentation(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """High-level wrapper for notebook calls."""
    csv_dir = Path(str(inputs.get("csv_dir", "")).strip()).expanduser()
    csv_file = str(inputs.get("csv_file", "")).strip()
    if not csv_dir or not csv_file:
        raise ValueError("run_data_segmentation requires 'csv_dir' and 'csv_file'.")

    csv_path = (csv_dir / csv_file).resolve()
    output_csv_file = str(inputs.get("output_csv_file", "") or "").strip()
    output_csv_path = (csv_dir / output_csv_file).resolve() if output_csv_file else None

    out_path = run_segmentation_pipeline(
        csv_path=csv_path,
        mutation_separator=str(inputs.get("mutation_separator", ":")).strip() or ":",
        num_mutation_segments_singlemut=int(inputs.get("num_mutation_segments_singlemut", 5)),
        min_layer_size_for_multimut_segmentation=int(
            inputs.get("min_layer_size_for_multimut_segmentation", 1000)
        ),
        max_layer_size_for_multimut_segmentation=_parse_optional_int_or_list(
            inputs.get("max_layer_size_for_multimut_segmentation", None)
        ),
        smallest_single_mutant_size=int(inputs.get("smallest_single_mutant_size", 100)),
        k_folds=int(inputs.get("k_folds", 5)),
        include_mutation_onehot_for_clustering=bool(
            inputs.get("include_mutation_onehot_for_clustering", True)
        ),
        output_csv_path=output_csv_path,
        verbose=bool(inputs.get("verbose", True)),
        print_group_details=bool(inputs.get("print_group_details", False)),
    )
    return {
        "status": "ok",
        "input_csv_path": str(csv_path),
        "output_csv_path": str(out_path),
        "output_dataset_fname": out_path.name,
    }


if __name__ == "__main__":
    # Sensible defaults for easy local editing in IDE runs (can be overridden by CLI args).
    RUN_DEFAULTS = {
        "csv_dir": "expdata/Q65J43_BACLD_g4_Thomas_2025",
        "csv_file": "Q65J43_BACLD_g4_Thomas_2025.csv",
        "mutation_separator": ":",
        "num_mutation_segments_singlemut": 4,
        "min_layer_size_for_multimut_segmentation": 1500,
        "max_layer_size_for_multimut_segmentation": None,
        "smallest_single_mutant_size": 60,
        "k_folds": 5,
        "include_mutation_onehot_for_clustering": True,
    }
    main(defaults=RUN_DEFAULTS)
