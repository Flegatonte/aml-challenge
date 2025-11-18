# src/step1_cosine_baseline.py
# baseline retrieval experiment using cosine similarity.
# this step provides a reference point before any learned mapping
# (i.e., before ridge or residual mlp). the idea is:
#   - compare text embeddings and image embeddings directly,
#   - align dimensions via pca if the two modalities live in different spaces,
#   - evaluate retrieval metrics (mrr, recall@k).
#
# this mirrors the "no adapter" baseline used in cross-modal retrieval papers, where cosine similarity is typically the main scoring 
# function.

import os, sys
from pathlib import Path
import numpy as np
import torch

# small cuda optimizations for more stable perf on large matrices.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# import cfg and project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import CFG
from utils import load_train_npz, train_val_split, l2_normalize
from metrics import mrr_full, recall_at_k_full

device = torch.device(CFG.STEP1.DEVICE if torch.cuda.is_available() else "cpu")
print("Using device:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

def sanitize(a: np.ndarray) -> np.ndarray:
    # numerical cleanup: replace nan/inf which can appear in rare cases
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)

def main():
    np.random.seed(CFG.SEED)

    # load dataset (text embeddings X, image embeddings Y)
    # lab[i] tells which image corresponds to caption i.
    X, Y, lab = load_train_npz(CFG.PATHS.TRAIN_NPZ)
    print(f"loaded: X{X.shape}  Y{Y.shape}")

    # split into train/val partitions (classical validation split).
    idx_tr, idx_val = train_val_split(
        X.shape[0],
        val_frac=float(CFG.STEP1.VAL_FRAC),
        seed=CFG.SEED
    )
    gt_train = np.unique(lab[idx_tr])  # all image indices seen in training part
    gt_val   = lab[idx_val]           # ground-truth image index per validation caption

    # ensure both sides have finite numeric values
    X_val = sanitize(X[idx_val])
    Y_all = sanitize(Y)

    dx, dy = X_val.shape[1], Y_all.shape[1]
    target_d = min(dx, dy)

    # dimensionality alignment via pca
    #
    # text and image embeddings may have different dimensionality. since cosine similarity requires both vectors to live 
    # in the same space, we reduce the higher-dimensional modality via pca.
    #
    # this is a standard trick in multimodal retrieval when comparing embeddings from independent encoders (similar to the 
    # preprocessing used in cca/pca baselines in cross-modal literature).

    if dx != dy:
        from sklearn.decomposition import PCA
        if dy > dx:
            # shrink image embeddings → dimension of text embeddings
            print(f"[PCA] reducing Y: {dy} -> {dx} (fit on {len(gt_train)} train images)")
            pcaY = PCA(n_components=dx, svd_solver="full", random_state=CFG.SEED)
            Y_fit = sanitize(Y[gt_train])
            pcaY.fit(Y_fit)
            Y_all = pcaY.transform(Y_all).astype("float32", copy=False)
        else:
            # shrink text embeddings → dimension of image embeddings
            print(f"[PCA] reducing X_val: {dx} -> {dy} (fit on {len(idx_tr)} train rows)")
            pcaX = PCA(n_components=dy, svd_solver="full", random_state=CFG.SEED)
            X_fit = sanitize(X[idx_tr])
            pcaX.fit(X_fit)
            X_val = pcaX.transform(X_val).astype("float32", copy=False)

    # normalize to unit vectors
    #
    # cosine similarity is essentially dot(x̂, ŷ).
    # l2-normalization ensures that retrieval reflects pure angular alignment rather than vector norm artifacts.

    Q = l2_normalize(X_val)
    G = l2_normalize(Y_all)

    # compute baseline retrieval metrics
    #
    # mrr_full: mean reciprocal rank across all validation queries.
    # recall@k: fraction of queries whose correct image appears in top-k.
    # these are the same quantities computed later on more sophisticated models.

    batch_eval = int(CFG.STEP1.EVAL_BATCH)
    mrr = mrr_full(Q, G, gt_val, batch=batch_eval)
    r1  = recall_at_k_full(Q, G, gt_val, k=1,  batch=batch_eval)
    r5  = recall_at_k_full(Q, G, gt_val, k=5,  batch=batch_eval)
    r10 = recall_at_k_full(Q, G, gt_val, k=10, batch=batch_eval)

    print(f"[COSINE] MRR={mrr:.4f} | R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}")

    # store metrics for reproducibility
    out_dir = Path(CFG.STEP1.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cosine_baseline.txt", "w") as f:
        f.write(f"MRR={mrr:.6f}\nR@1={r1:.6f}\nR@5={r5:.6f}\nR@10={r10:.6f}\n")

if __name__ == "__main__":
    main()
