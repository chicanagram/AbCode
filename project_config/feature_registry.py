from __future__ import annotations

# Default feature-set selections for notebook-driven sequence encoding runs.
# Keep these editable outside notebooks when the list gets long.

# --- Classical feature sets ---
CLASSICAL_ENCODING_FEATURE_SETS = [
    "one_hot",
    "georgiev",
    "one_hot_per_residue",
    "georgiev_per_residue",
    "one_hot_mean_pooled",
    "georgiev_mean_pooled",
]

PHYSICOCHEMICAL_ENCODING_FEATURE_SETS = [
    'length',
    'aac',
    'aaindex1',
    'ctdc',
    'ctdt',
]

# --- PLM feature sets ---
PLM_MODELS_DICT = {
    "esm2-650m": [33],
    "esmc-600m": [36],
    "poet2": [12],
}

PLM_ENCODING_FEATURE_SETS = [
    f
    for f_list in [
        [f"{plm}_LLR", f"{plm}_PLL", f"{plm}_PLL_mean_pooled", f"{plm}_per_residue", f"{plm}_mean_pooled", f"{plm}_mut_pooled", f"{plm}_svd_pooled"]
        for plm in PLM_MODELS_DICT
    ]
    for f in f_list
]

# combination of classical and PLM
FEATURE_SETS_ALL = CLASSICAL_ENCODING_FEATURE_SETS + PLM_ENCODING_FEATURE_SETS
FEATURE_SETS_DEFAULT = [
    "one_hot",
    "esm2-650m_LLR-masked",
    "esmc-600m_LLR-masked",
    "esm2-650m_PLL-masked",
    "esmc-600m_PLL-masked",
    "esm2-650m_mean_pooled",
    "esmc-600m_mean_pooled",
]

# combi feature sets
COMBI_ML_FEATURE_SETS = {
    'onehot': ['one_hot'],
    'georgiev': ['georgiev'],
    'onehot_esm2_LLR': ['one_hot', 'esm2-650m_LLR-masked'],
    'onehot_esmc_LLR': ['one_hot', 'esmc-600m_LLR-masked'],
    'georgiev_esm2_LLR': ['georgiev', 'esm2-650m_LLR-masked'],
    'georgiev_esmc_LLR': ['georgiev', 'esmc-600m_LLR-masked'],
    'esm2_seq_embeddings': ['esm2-650m_mean_pooled-33'],
    'esmc_seq_embeddings': ['esmc-600m_mean_pooled-36'],
    'esm2_seq_embeddings_LLR': ['esm2-650m_mean_pooled-33', 'esm2-650m_LLR-masked'],
    'esmc_seq_embeddings_LLR': ['esmc-600m_mean_pooled-36', 'esmc-600m_LLR-masked'],
}
