import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import mean_squared_error

from .config import cfg
from .model import CrossAttentionFusionModel
from .dataloader import prepare_kfold_generator, prepare_final_dataloaders
from .evaluate import evaluate_performance, visualize_attention


def train_and_validate_one_fold(model, fold_idx, fold_package, fold_log_dir):
    writer = SummaryWriter(log_dir=fold_log_dir)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.MIN_LR)
    use_bf16_runtime = cfg.USE_BF16 and cfg.DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()

    best_val_loss_in_fold = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(fold_log_dir, "best_model.pth")

    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        pbar_desc = f"Trait {cfg.TARGET_TRAIT_INDEX} [Fold {fold_idx + 1}] Ep {epoch + 1}"
        pbar = tqdm(fold_package["train_loader"], desc=pbar_desc, leave=False)
        train_loss_sum = 0
        for snp, grm, target in pbar:
            snp, grm, target = snp.to(cfg.DEVICE), grm.to(cfg.DEVICE), target.to(cfg.DEVICE).view(-1, 1)
            with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=torch.bfloat16, enabled=use_bf16_runtime):
                preds = model(snp, grm)
                loss = criterion(preds, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * snp.size(0)
            pbar.set_postfix({'train_loss': loss.item()})
        avg_train_loss = train_loss_sum / len(fold_package["train_loader"].dataset)
        writer.add_scalar('Loss/Train_MSE_Normalized', avg_train_loss, epoch)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for snp, grm, target in fold_package["val_loader"]:
                snp, grm, target = snp.to(cfg.DEVICE), grm.to(cfg.DEVICE), target.to(cfg.DEVICE).view(-1, 1)
                with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=torch.bfloat16, enabled=use_bf16_runtime):
                    outputs = model(snp, grm)
                val_loss_sum += criterion(outputs, target).item() * snp.size(0)
        avg_val_loss = val_loss_sum / len(fold_package["val_loader"].dataset)
        writer.add_scalar('Loss/Validation_MSE_Normalized', avg_val_loss, epoch)
        scheduler.step()

        if avg_val_loss < best_val_loss_in_fold:
            best_val_loss_in_fold = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if cfg.USE_EARLY_STOPPING and epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"--- [Early Stopping on Fold {fold_idx + 1}] Stopped at epoch {epoch + 1}. ---")
            break

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    val_preds_norm_final = []
    with torch.no_grad():
        for snp, grm, target in fold_package["val_loader"]:
            snp, grm = snp.to(cfg.DEVICE), grm.to(cfg.DEVICE)
            with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=torch.bfloat16, enabled=use_bf16_runtime):
                outputs = model(snp, grm)
            val_preds_norm_final.extend(outputs.float().cpu().numpy())

    val_preds_orig_final = np.array(val_preds_norm_final) * fold_package["pheno_std"] + fold_package["pheno_mean"]
    final_fold_metrics = evaluate_performance(fold_package["pheno_val"], val_preds_orig_final)
    final_fold_metrics['mse_norm'] = best_val_loss_in_fold

    writer.add_hparams(
        {"fold": fold_idx + 1},
        {
            "hparam/best_mse_norm": final_fold_metrics['mse_norm'],
            "hparam/final_pearson_r": final_fold_metrics.get('r', np.nan),
            "hparam/final_r2": final_fold_metrics.get('R2', np.nan),
        },
        run_name='final_metrics'
    )
    writer.close()

    return final_fold_metrics


def execute_train_mode(snp_dev, pheno_dev, traits_to_run, top_level_log_dir):
    for trait_idx in traits_to_run:
        cfg.TARGET_TRAIT_INDEX = trait_idx
        trait_log_dir = os.path.join(top_level_log_dir, f"Trait_{trait_idx}")
        os.makedirs(trait_log_dir, exist_ok=True)
        print("\n" + "=" * 25 + f" Starting CV for Trait {trait_idx} " + "=" * 25)

        pheno_dev_trait = pheno_dev[:, trait_idx]
        data_generator = prepare_kfold_generator(snp_dev, pheno_dev_trait, trait_log_dir)

        all_fold_metrics = []
        for fold_idx, fold_package in data_generator:
            fold_log_dir = os.path.join(trait_log_dir, f"fold_{fold_idx + 1}")

            model = CrossAttentionFusionModel(
                num_snp_values=cfg.NUM_SNP_VALUES, grm_vector_len=fold_package["grm_train_dim"], num_outputs=1,
                embed_dim=cfg.VIT_EMBED_DIM, depth=cfg.VIT_DEPTH, num_heads=cfg.VIT_NUM_HEADS,
                mlp_ratio=cfg.VIT_MLP_RATIO, drop_ratio=cfg.VIT_DROP_RATIO, attn_drop_ratio=cfg.VIT_ATTN_DROP_RATIO,
                fusion_start_index=cfg.FUSION_START_INDEX, grm_mlp_hidden_layers=cfg.GRM_MLP_HIDDEN_LAYERS,
                max_relative_position=fold_package["max_rel_pos"], patch_size=fold_package["patch_size"],
                padding_idx=cfg.PADDING_IDX
            ).to(cfg.DEVICE)

            fold_metrics = train_and_validate_one_fold(model, fold_idx, fold_package, fold_log_dir)
            all_fold_metrics.append(fold_metrics)

        summary_path = os.path.join(trait_log_dir, "cv_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Cross-Validation Summary for Trait {trait_idx}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Individual Fold Performance:\n")
            for i, metrics in enumerate(all_fold_metrics):
                r_val = metrics.get('r', 0.0)
                r2_val = metrics.get('R2', 0.0)
                mse_norm_val = metrics.get('mse_norm', 0.0)
                f.write(f"  Fold {i + 1}: Pearson's r={r_val:.4f}, R2={r2_val:.4f}, MSE_Norm={mse_norm_val:.5f}\n")
            f.write("\n" + "=" * 50 + "\n\n")

            f.write("Average Performance:\n")
            if all_fold_metrics:
                avg_r = np.nanmean([m['r'] for m in all_fold_metrics])
                std_r = np.nanstd([m['r'] for m in all_fold_metrics])
                avg_r2 = np.nanmean([m['R2'] for m in all_fold_metrics])
                std_r2 = np.nanstd([m['R2'] for m in all_fold_metrics])
                avg_mse_norm = np.nanmean([m['mse_norm'] for m in all_fold_metrics])
                std_mse_norm = np.nanstd([m['mse_norm'] for m in all_fold_metrics])
                f.write(f"  Average Pearson's r      : {avg_r:.4f} ± {std_r:.4f}\n")
                f.write(f"  Average R-squared (R2)   : {avg_r2:.4f} ± {std_r2:.4f}\n")
                f.write(f"  Average MSE (Normalized) : {avg_mse_norm:.5f} ± {std_mse_norm:.5f}\n")
            else:
                f.write("No folds were successfully trained.\n")
        print(f"--- CV summary saved to {summary_path} ---")


def execute_test_mode(snp_dev, pheno_dev, snp_test, pheno_test, traits_to_run, top_level_log_dir):

    for trait_idx in traits_to_run:
        cfg.TARGET_TRAIT_INDEX = trait_idx
        trait_log_dir = os.path.join(top_level_log_dir, f"Trait_{trait_idx}")
        os.makedirs(trait_log_dir, exist_ok=True)

        pheno_dev_trait = pheno_dev[:, trait_idx]
        pheno_test_trait = pheno_test[:, trait_idx]

        data_package = prepare_final_dataloaders(snp_dev, pheno_dev_trait, snp_test, pheno_test_trait, trait_log_dir)
        if not data_package: continue

        final_model = CrossAttentionFusionModel(
            num_snp_values=cfg.NUM_SNP_VALUES, grm_vector_len=data_package["grm_train_dim"], num_outputs=1,
            embed_dim=cfg.VIT_EMBED_DIM, depth=cfg.VIT_DEPTH, num_heads=cfg.VIT_NUM_HEADS,
            mlp_ratio=cfg.VIT_MLP_RATIO, drop_ratio=cfg.VIT_DROP_RATIO, attn_drop_ratio=cfg.VIT_ATTN_DROP_RATIO,
            fusion_start_index=cfg.FUSION_START_INDEX, grm_mlp_hidden_layers=cfg.GRM_MLP_HIDDEN_LAYERS,
            max_relative_position=data_package["max_rel_pos"], patch_size=data_package["patch_size"],
            padding_idx=cfg.PADDING_IDX
        ).to(cfg.DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(final_model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.MIN_LR)
        use_bf16_runtime = cfg.USE_BF16 and cfg.DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()
        writer = SummaryWriter(log_dir=trait_log_dir)

        for epoch in range(cfg.NUM_EPOCHS):
            final_model.train()
            pbar_desc = f"Trait {trait_idx} [Final Train] Ep {epoch + 1}"
            pbar = tqdm(data_package["train_loader"], desc=pbar_desc, leave=False)
            for snp, grm, target in pbar:
                snp, grm, target = snp.to(cfg.DEVICE), grm.to(cfg.DEVICE), target.to(cfg.DEVICE).view(-1, 1)
                with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=torch.bfloat16, enabled=use_bf16_runtime):
                    preds = final_model(snp, grm)
                    loss = criterion(preds, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({'dev_loss': loss.item()})
            scheduler.step()
            writer.add_scalar('Loss/Final_Dev_Train_MSE_Normalized', loss.item(), epoch)

        print("--- [Final Evaluation] Training complete. Evaluating final model on the hold-out test set... ---")

        final_model.eval()
        test_preds_norm = []
        with torch.no_grad():
            for snp, grm, _ in data_package["test_loader"]:
                snp, grm = snp.to(cfg.DEVICE), grm.to(cfg.DEVICE)
                with torch.amp.autocast(device_type=cfg.DEVICE.type, dtype=torch.bfloat16, enabled=use_bf16_runtime):
                    outputs = final_model(snp, grm)
                test_preds_norm.extend(outputs.float().cpu().numpy())

        test_preds_orig = np.array(test_preds_norm) * data_package["pheno_std"] + data_package["pheno_mean"]
        final_test_metrics = evaluate_performance(pheno_test_trait, test_preds_orig)

        pheno_test_norm = data_package["pheno_test_norm"]
        mse_norm = mean_squared_error(pheno_test_norm, test_preds_norm)
        final_test_metrics['mse_norm'] = mse_norm

        print(f"--- Final Test Metrics for Trait {trait_idx} ---")
        print(f"  Pearson's r: {final_test_metrics.get('r', 'N/A'):.4f}")
        print(f"  R-squared (R2): {final_test_metrics.get('R2', 'N/A'):.4f}")
        print(f"  MSE (Normalized): {final_test_metrics.get('mse_norm', 'N/A'):.5f}")

        summary_path = os.path.join(trait_log_dir, "test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Final Test Summary for Trait {trait_idx}\n")
            f.write("=" * 50 + "\n")
            f.write(f"  Pearson's r      : {final_test_metrics.get('r', np.nan):.4f}\n")
            f.write(f"  R-squared (R2)   : {final_test_metrics.get('R2', np.nan):.4f}\n")
            f.write(f"  MSE (Normalized) : {final_test_metrics.get('mse_norm', np.nan):.5f}\n")
        print(f"--- Test summary saved to {summary_path} ---")

        writer.add_hparams(
            {"trait_index": trait_idx},
            {
                "hparam/final_pearson_r": final_test_metrics.get('r', np.nan),
                "hparam/final_r2": final_test_metrics.get('R2', np.nan),
                "hparam/final_mse_norm": final_test_metrics.get('mse_norm', np.nan)
            },
            run_name='final_test_metrics'
        )

        visualize_attention(final_model, data_package["test_loader"], cfg.DEVICE, trait_log_dir)
        writer.close()