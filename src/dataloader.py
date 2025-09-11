# path: src/dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
from sklearn.model_selection import KFold

from .config import cfg
from .utils import select_features_with_lgbm, calculate_grm_vanraden


class SNPGRMDataset(Dataset):
    def __init__(self, snp_tensor, grm_tensor, targets_tensor):
        assert len(snp_tensor) == len(grm_tensor) == len(targets_tensor)
        self.snp_tensor, self.grm_tensor, self.targets_tensor = snp_tensor, grm_tensor, targets_tensor

    def __len__(self): return len(self.snp_tensor)

    def __getitem__(self, idx): return self.snp_tensor[idx], self.grm_tensor[idx], self.targets_tensor[idx]


def load_full_dataset():

    dataset_path = os.path.join(cfg.DATA_DIR, cfg.DATASET_NAME)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    with np.load(dataset_path) as f:
        all_snp = f['geno'].astype(np.int64)
        all_pheno = f['pheno'].astype(np.float32)
    all_snp[all_snp == -1] = cfg.PADDING_IDX
    if all_pheno.ndim == 1: all_pheno = all_pheno.reshape(-1, 1)
    return all_snp, all_pheno


def prepare_kfold_generator(snp_dev, pheno_dev, trait_log_dir):
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(kf.split(snp_dev)):
        print(f"\n--- [Fold {fold_idx + 1}/5] Preparing data... ---")
        snp_train_fold, snp_val_fold = snp_dev[train_fold_idx], snp_dev[val_fold_idx]
        pheno_train_fold, pheno_val_fold = pheno_dev[train_fold_idx], pheno_dev[val_fold_idx]

        grm_train_fold = calculate_grm_vanraden(snp_train_fold, snp_train_fold)
        grm_val_fold = calculate_grm_vanraden(snp_val_fold, snp_train_fold)

        fold_log_dir = os.path.join(trait_log_dir, f"fold_{fold_idx + 1}")
        os.makedirs(fold_log_dir, exist_ok=True)
        snp_train_selected, sel_indices = select_features_with_lgbm(snp_train_fold, pheno_train_fold, fold_log_dir)
        if snp_train_selected.shape[1] == 0:
            print(f"--- [Error] Fold {fold_idx + 1}: No effective features selected. Skipping fold. ---");
            continue
        snp_val_selected = snp_val_fold[:, sel_indices]

        mean, std = pheno_train_fold.mean(), pheno_train_fold.std()
        std = 1.0 if std < 1e-8 else std
        pheno_train_norm, pheno_val_norm = (pheno_train_fold - mean) / std, (pheno_val_fold - mean) / std

        train_ds = SNPGRMDataset(torch.from_numpy(snp_train_selected).long(), torch.from_numpy(grm_train_fold).float(),
                                 torch.from_numpy(pheno_train_norm).float())
        val_ds = SNPGRMDataset(torch.from_numpy(snp_val_selected).long(), torch.from_numpy(grm_val_fold).float(),
                               torch.from_numpy(pheno_val_norm).float())
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

        patch_size = math.ceil(snp_train_selected.shape[1] / cfg.NUM_PATCHES)

        yield fold_idx, {
            "train_loader": train_loader, "val_loader": val_loader, "pheno_val": pheno_val_fold,
            "pheno_mean": mean, "pheno_std": std, "grm_train_dim": grm_train_fold.shape[1],
            "patch_size": patch_size, "max_rel_pos": cfg.NUM_PATCHES
        }


def prepare_final_dataloaders(snp_dev, pheno_dev, snp_test, pheno_test, trait_log_dir):
    grm_dev = calculate_grm_vanraden(snp_dev, snp_dev)
    grm_test = calculate_grm_vanraden(snp_test, snp_dev)

    snp_dev_selected, sel_indices = select_features_with_lgbm(snp_dev, pheno_dev, trait_log_dir)
    if snp_dev_selected.shape[1] == 0:
        return None
    snp_test_selected = snp_test[:, sel_indices]

    mean, std = pheno_dev.mean(), pheno_dev.std()
    std = 1.0 if std < 1e-8 else std

    pheno_dev_norm = (pheno_dev - mean) / std
    pheno_test_norm = (pheno_test - mean) / std

    train_ds = SNPGRMDataset(torch.from_numpy(snp_dev_selected).long(), torch.from_numpy(grm_dev).float(),
                             torch.from_numpy(pheno_dev_norm).float())
    test_ds = SNPGRMDataset(torch.from_numpy(snp_test_selected).long(), torch.from_numpy(grm_test).float(),
                            torch.from_numpy(np.zeros_like(pheno_test)).float())  # Target仍然是占位符
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    patch_size = math.ceil(snp_dev_selected.shape[1] / cfg.NUM_PATCHES)

    return {
        "train_loader": train_loader, "test_loader": test_loader,
        "pheno_test": pheno_test,
        "pheno_test_norm": pheno_test_norm,
        "pheno_mean": mean, "pheno_std": std, "grm_train_dim": grm_dev.shape[1],
        "patch_size": patch_size, "max_rel_pos": cfg.NUM_PATCHES
    }