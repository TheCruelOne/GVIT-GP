import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from .config import cfg


def evaluate_performance(y_true, y_pred):

    if not np.all(np.isfinite(y_pred)):
        return {"r": np.nan, "R2": np.nan}

    y_true_flat = np.ravel(y_true)
    y_pred_flat = np.ravel(y_pred)

    if np.std(y_true_flat) < 1e-6:
        r, r2 = np.nan, np.nan
    else:
        if np.std(y_pred_flat) < 1e-6:
            r = 0.0
        else:
            r, _ = pearsonr(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

    return {"r": r, "R2": r2}


def visualize_attention(model, data_loader, device, log_dir):

    model.eval()
    snp, grm, _ = next(iter(data_loader))

    snp_sample = snp[0:1].to(device)
    grm_sample = grm[0:1].to(device)

    with torch.no_grad():
        _, attention = model(snp_sample, grm_sample, return_attention=True)

    avg_attention = attention.mean(dim=1).squeeze(0).cpu().numpy()
    cls_attention = avg_attention[0, :]

    plt.figure(figsize=(15, 2))
    sns.heatmap([cls_attention], cmap='viridis', cbar=True)
    plt.title(f'Attention from [CLS] Token to Patches (Trait {cfg.TARGET_TRAIT_INDEX})')
    plt.xlabel('Token Index (0=[CLS], 1...N=Patches)')
    plt.ylabel('Attention Source ([CLS])')
    plt.yticks([])

    save_path = os.path.join(log_dir, f"attention_map_cls_trait_{cfg.TARGET_TRAIT_INDEX}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()