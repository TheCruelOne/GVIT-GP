import torch


class Config:

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_LOG_DIR = './logs/'

    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 30
    USE_BF16 = True
    MIN_LR = 1e-7

    NUM_PATCHES = 800
    NUM_SNP_VALUES = 4
    PADDING_IDX = 3
    VIT_EMBED_DIM = 192
    VIT_DEPTH = 6
    VIT_NUM_HEADS = 4
    VIT_MLP_RATIO = 4.0
    VIT_DROP_RATIO = 0.2
    VIT_ATTN_DROP_RATIO = 0.
    FUSION_START_INDEX = 0
    GRM_MLP_HIDDEN_LAYERS = [256, 32]


cfg = Config()