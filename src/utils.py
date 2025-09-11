import torch
import numpy as np
import random
import os
import lightgbm as lgb
from .config import cfg

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_grm_vanraden(target_geno, base_geno):
    num_markers = base_geno.shape[1]
    if num_markers == 0: return np.zeros((target_geno.shape[0], base_geno.shape[0]), dtype=np.float32)
    p = np.mean(base_geno, axis=0) / 2.0
    denominator = 2 * np.sum(p * (1 - p))
    if denominator == 0: return np.zeros((target_geno.shape[0], base_geno.shape[0]), dtype=np.float32)
    P = 2 * p
    Z_target, Z_base = target_geno - P, base_geno - P
    grm = (Z_target @ Z_base.T) / denominator
    return grm.astype(np.float32)

def select_features_with_lgbm(snp_train, pheno_train, log_dir):
    lgbm = lgb.LGBMRegressor(random_state=cfg.SEED, n_jobs=-1)
    lgbm.fit(snp_train, pheno_train)
    importances = lgbm.feature_importances_
    original_indices = np.where(importances > 0)[0]

    is_fold_dir = "fold" in os.path.basename(log_dir) or "fold" in os.path.basename(os.path.dirname(log_dir))
    filename = "selected_features.txt" if is_fold_dir else "selected_features_final.txt"
    indices_save_path = os.path.join(log_dir, filename)

    if len(original_indices) == 0:
        print("Warning: No features with importance > 0 were found.")
        np.savetxt(indices_save_path, [], fmt='%d')
        return np.array([[]]), np.array([])

    np.savetxt(indices_save_path, original_indices, fmt='%d')
    print(f"--- Selected {len(original_indices)} features. Indices saved to {indices_save_path} ---")
    return snp_train[:, original_indices], original_indices