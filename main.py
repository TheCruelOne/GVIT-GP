import argparse
import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from src.config import cfg
from src.utils import set_seed
from src.dataloader import load_full_dataset
from src.train import execute_train_mode, execute_test_mode


def main():
    parser = argparse.ArgumentParser(description="Genomic Fusion Model Training and Testing")

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help="Set the operational mode: 'train' for cross-validation on a dev set, 'test' for final training and evaluation.")
    parser.add_argument('--data_dir', type=str, default='./datasets/',
                        help='Directory containing the datasets')
    parser.add_argument('--dataset', type=str, default='soybean15899',
                        help='Dataset name without the .npz extension')
    parser.add_argument('--run_id', type=str,
                        help="Required for 'test' mode. The timestamp ID of the 'train' run to use for final evaluation.")

    parser.add_argument('--trait_indices', type=int, nargs='*', default=None,
                        help='Space-separated list of trait indices to process. If not provided, loops through all available traits.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=2027, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')

    args = parser.parse_args()

    if args.mode == 'test' and not args.run_id:
        parser.error("--run_id is required when --mode is 'test'")

    cfg.DATA_DIR = args.data_dir
    cfg.DATASET = args.dataset
    cfg.DATASET_NAME = cfg.DATASET + '.npz'
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LEARNING_RATE = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.SEED = args.seed
    cfg.NUM_WORKERS = args.num_workers

    set_seed(cfg.SEED)

    run_id_to_use = args.run_id if args.mode == 'test' else datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"{run_id_to_use}_{args.mode}_seed{cfg.SEED}_lr{cfg.LEARNING_RATE}_{cfg.DATASET}"
    top_level_log_dir = os.path.join(cfg.BASE_LOG_DIR, log_dir_name)
    os.makedirs(top_level_log_dir, exist_ok=True)
    print(f"--- [Log] Main log directory for this run: {top_level_log_dir} ---")

    all_snp, all_pheno_multi = load_full_dataset()
    rows_with_any_nan = np.any(np.isnan(all_pheno_multi), axis=1)
    snp_cleaned = all_snp[~rows_with_any_nan]
    pheno_cleaned = all_pheno_multi[~rows_with_any_nan]


    num_available_traits = pheno_cleaned.shape[1]
    if args.trait_indices:
        traits_to_run = [i for i in args.trait_indices if 0 <= i < num_available_traits]
        print(f"--- Will process specified trait indices: {traits_to_run} ---")
    else:
        traits_to_run = range(num_available_traits)
        print(f"--- Will loop through all {num_available_traits} available traits. ---")

    if args.mode == 'train':
        dev_indices, test_indices = train_test_split(
            np.arange(len(snp_cleaned)), test_size=0.2, random_state=cfg.SEED
        )
        test_indices_path = os.path.join(top_level_log_dir, "test_indices.txt")
        np.savetxt(test_indices_path, test_indices, fmt='%d')

        snp_dev, pheno_dev = snp_cleaned[dev_indices], pheno_cleaned[dev_indices]
        execute_train_mode(snp_dev, pheno_dev, traits_to_run, top_level_log_dir)

    elif args.mode == 'test':
        train_log_dir_name = f"{args.run_id}_train_seed{cfg.SEED}_lr{cfg.LEARNING_RATE}_{cfg.DATASET}"
        train_log_dir = os.path.join(cfg.BASE_LOG_DIR, train_log_dir_name)
        test_indices_path = os.path.join(train_log_dir, "test_indices.txt")

        if not os.path.exists(test_indices_path):
            raise FileNotFoundError(
                f"Could not find test_indices.txt in the specified train run directory: {train_log_dir}")

        test_indices = np.loadtxt(test_indices_path, dtype=int)
        dev_indices = np.setdiff1d(np.arange(len(snp_cleaned)), test_indices)

        snp_dev, pheno_dev = snp_cleaned[dev_indices], pheno_cleaned[dev_indices]
        snp_test, pheno_test = snp_cleaned[test_indices], pheno_cleaned[test_indices]

        execute_test_mode(snp_dev, pheno_dev, snp_test, pheno_test, traits_to_run, top_level_log_dir)


if __name__ == '__main__':
    main()