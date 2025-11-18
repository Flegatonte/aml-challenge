# src/step2_affine.py
# the goal of this step is to:
#   learn a simple affine map T(x) = x W^T + b that takes caption embeddings into the image embedding space.
#
# overview:
#   - we keep both encoders frozen (as in the challenge description).
#   - we only learn the "stitching" between the two latent spaces.
#   - we are trying to replicate the idea presented in the following papers:
#       * "Latent Space Translation via Semantic Alignment" (Maiorca et al., NeurIPS 2023)
#       * "vec2vec: Harnessing the Universal Geometry of Embeddings"
#     where a linear / affine map is often enough if embeddings are normalized properly.

import os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

# QoL / CUDA setup:
# a few quality-of-life flags for running on GPU:
# - expandable_segments: lets PyTorch grow GPU memory segments instead of failing with fragmentation errors on long runs.
# - cudnn.benchmark: enables autotuning of convolution algorithms based on the input shapes (safe here because shapes are basically fixed).
# - high matmul precision: slightly more accurate float32 matmuls on recent GPUs.

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

# project config & imports:
# instead of hard-coding paths / hyperparams or passing everything via argparse,
# we centralize them in a single config object (CFG). this is mainly for sanity:
# - STEP2.* can store things like epochs, batch_size, ridge_alpha, etc.
# - PATHS.* can store the standard locations for NPZ data and artifact folders.
#
# the aim is to make it easier to keep the same code running on local / cluster machines just by changing one config file.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from config import CFG  # expected to define STEP2.* and PATHS.*

from utils import (
    load_train_npz,
    train_val_split,
    fit_scalers,
    apply_scaler,
    l2_normalize,
)
from metrics import mrr_full

# numeric hygiene:
def sanitize(a: np.ndarray) -> np.ndarray:
    """
    basic numeric cleanup for numpy arrays.

    in practice:
    - replaces NaN and +/-inf with 0.0
    - returns float32 (the dtype we use for most matmuls)

    this is a defensive step to make sure weird values do not propagate into:
    - the closed-form ridge solve (which uses SVD),
    - the PyTorch forward passes (where NaNs would kill the loss).
    """
    return np.nan_to_num(
        a,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype("float32", copy=False)


# models:
class AffineProjector(nn.Module):
    """
    basic affine map:
        y_hat = x W^T + b

    where:
        x      ∈ R^{d_in}
        y_hat  ∈ R^{d_out}

    this is the main object we care about: it implements the linear/affine stitching between text and image space, 
    like in the "latent space translation" / vec2vec style papers.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        # start with zero bias so that the ridge solution fully controls
        # the initial map (we can learn a small shift later if needed).
        with torch.no_grad():
            self.lin.bias.zero_()

    @torch.no_grad()
    def init_from_W(self, W_numpy: np.ndarray) -> None:
        """
        load a closed-form ridge solution into the layer.

        W_numpy is stored as [d_in, d_out] (numpy convention), while PyTorch expects [out_features, in_features],
        so we just transpose before copying.
        """
        assert W_numpy.shape == (self.lin.in_features, self.lin.out_features)
        W_t = torch.from_numpy(W_numpy.T.astype("float32", copy=False))
        self.lin.weight.copy_(W_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class ResidualMLPProjector(nn.Module):
    """
    small residual variant on top of the affine map.

    idea:
      - start from the same affine map as above (initialized from ridge);
      - let a tiny MLP learn a residual correction in the *image* space:

          y0   = x W^T + b
          y_ml = MLP( LN(y0) )
          ŷ    = y0 + y_ml

    the last linear layer in the MLP is zero-initialized so that, at the beginning of training, the model behaves exactly like the 
    pure affine projector. this keeps the comparison "fair" and avoids wild drifts.
    """

    def __init__(
        self,
        d_in: int, # dimension of the input (text embedding).  
        d_out: int, # dimension of the output (image embedding space).  
        hidden: int = 1024, # width of the hidden layer inside each MLP block.  
        depth: int = 2, # number of residual MLP blocks stacked sequentially.  
        dropout: float = 0.1, # dropout probability used inside each block to avoid overfitting and stabilize training.
    ):
        super().__init__()

        # base affine part (same role as AffineProjector.lin)
        self.base = nn.Linear(d_in, d_out)
        with torch.no_grad():
            self.base.bias.zero_()

        # small residual MLP that works in image space (dimension d_out)
        blocks: list[nn.Module] = []
        dim = d_out
        for _ in range(depth):
            # each block:
            #  LN(y) → Linear(d_out → hidden) → GELU → Dropout → Linear(hidden → d_out)
            #
            # meaning:
            #   - LayerNorm stabilizes scale and shifts of y0
            #   - Linear → GELU creates a non-linear transformation
            #   - Dropout adds regularization
            #   - final Linear projects back to d_out so the correction Δy
            #     can be added to y0 (residual connection)
            blocks.extend(
                [
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, dim),
                ]
            )
        self.res = nn.Sequential(*blocks)

        # zero-init only the very last Linear → start as pure affine
        for module in reversed(self.res):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break  # only touch the last Linear

    @torch.no_grad()
    def init_from_W(self, W_numpy: np.ndarray) -> None:
        """
        initialize the base affine part from the same ridge solution used in AffineProjector. 
        the residual MLP starts as a no-op.
        """
        assert W_numpy.shape == (self.base.in_features, self.base.out_features)
        W_t = torch.from_numpy(W_numpy.T.astype("float32", copy=False))
        self.base.weight.copy_(W_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # first go through the affine map, then apply a small residual
        y0 = self.base(x)
        return y0 + self.res(y0)


def main():
    # config / paths
    # we keep most knobs in a central CFG so that:
    #   - running on laptop vs. GPU box is just a config change
    #   - step2 and step3 share the same paths and seed
    train_npz = CFG.PATHS.TRAIN_NPZ
    artifacts_root = Path(CFG.PATHS.ARTIFACTS)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # hyper-parameters for this step (affine + optional residual MLP)
    val_frac     = float(getattr(CFG.STEP2, "VAL_FRAC"))
    ridge_alpha  = float(getattr(CFG.STEP2, "RIDGE_ALPHA"))
    epochs       = int(getattr(CFG.STEP2, "EPOCHS"))
    batch_size   = int(getattr(CFG.STEP2, "BATCH_SIZE"))
    lr           = float(getattr(CFG.STEP2, "LR"))
    weight_decay = float(getattr(CFG.STEP2, "WEIGHT_DECAY"))
    eval_batch   = int(getattr(CFG.STEP2, "EVAL_BATCH"))

    dev_str   = str(getattr(CFG.STEP2, "DEVICE", "cuda"))
    head_type = str(getattr(CFG.STEP2, "HEAD", "resmlp")).lower()

    # MLP-specific params (only used if HEAD == "resmlp")
    mlp_hidden  = int(getattr(CFG.STEP2, "MLP_HIDDEN"))
    mlp_depth   = int(getattr(CFG.STEP2, "MLP_LAYERS"))
    mlp_dropout = float(getattr(CFG.STEP2, "MLP_DROPOUT"))

    # seeding for reproducibility
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)

    dev = torch.device(dev_str if torch.cuda.is_available() else "cpu")

    # optional: organize artifacts by head type and ridge alpha
    if getattr(CFG.STEP2, "SAVE_RUN_SUBDIR", True):
        tag = f"{head_type}_alpha_{ridge_alpha:g}"
        out_dir = artifacts_root / f"affine_{tag}"
    else:
        out_dir = artifacts_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load caption/image embeddings and labels from npz
    #    X: [N_captions, d_text]
    #    Y: [N_images,   d_img]
    #    lab[i] = index of the correct image for caption i
    X, Y, lab = load_train_npz(train_npz)
    print(f"loaded: X{X.shape}  Y{Y.shape}")

    # 2) split captions into train / validation
    #    we always split over captions; images are retrieved via lab[]
    idx_tr, idx_val = train_val_split(
        X.shape[0],
        val_frac=val_frac,
        seed=CFG.SEED,
    )
    # training pairs: (caption embedding, its ground-truth image embedding)
    X_tr_raw, Y_tr_raw = X[idx_tr], Y[lab[idx_tr]]

    # validation captions + their target indices for full-gallery retrieval
    X_val_raw = X[idx_val]
    gt_val    = lab[idx_val]

    # 3) feature scaling + basic numeric cleanup
    #
    # following the vec2vec / semantic alignment literature, we:
    #   - standardize both domains (per-feature mean/std)
    #   - estimate the linear map in this normalized space
    #
    # this makes the geometry more comparable across modalities.

    sx, sy, X_tr_s, Y_tr_s = fit_scalers(X_tr_raw, Y_tr_raw)
    X_tr_s = sanitize(X_tr_s)
    Y_tr_s = sanitize(Y_tr_s)

    # 4) closed-form ridge regression (X_s → Y_s) in float64
    #
    #   argmin_W  || X_s W - Y_s ||^2 + α ||W||^2
    #
    # this is our "Procrustes-like" initialization:
    # with proper scaling and a small α, the solution tends to be close to an orthogonal map, as 
    # suggested in Maiorca et al. / vec2vec-style work.

    d_in, d_out = X.shape[1], Y.shape[1]
    print(f"fitting ridge map: {d_in} -> {d_out} (alpha={ridge_alpha})")

    X64 = X_tr_s.astype(np.float64, copy=False)
    Y64 = Y_tr_s.astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ridge = Ridge(
            alpha=ridge_alpha,
            fit_intercept=False,
            solver="svd",
            positive=False,
        )
        ridge.fit(X64, Y64)

    # sklearn stores coef_ as [d_out, d_in]; we want [d_in, d_out]
    W = ridge.coef_.T.astype("float32", copy=False)
    print("W ridge shape:", W.shape)

    # simple diagnostic: we check how far is W^T W from the identity
    with np.errstate(all="ignore"):
        ww = W.T @ W
        drift = float(
            np.linalg.norm(
                ww - np.eye(ww.shape[0], dtype=ww.dtype),
                ord="fro",
            )
        )
        drift_norm = drift / float(ww.shape[0])
    print(f"[ridge] orthogonality drift = {drift:.6f}  (norm={drift_norm:.6f})")

    # log hyper-params and diagnostics for this run
    with open(out_dir / "run_hparams.json", "w") as f:
        json.dump(
            {
                "head": head_type,
                "ridge_alpha": float(ridge_alpha),
                "orthogonality_drift": drift,
                "orthogonality_drift_norm": drift_norm,
                "val_frac": float(val_frac),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "eval_batch": int(eval_batch),
                "device": dev_str,
                "mlp_hidden": int(mlp_hidden),
                "mlp_depth": int(mlp_depth),
                "mlp_dropout": float(mlp_dropout),
            },
            f,
            indent=2,
        )

    # 5) pick the projector head:
    #    - plain AffineProjector: just the ridge map + fine-tuning
    #    - ResidualMLPProjector: affine + small non-linear residual in image space

    if head_type == "resmlp":
        net = ResidualMLPProjector(
            d_in,
            d_out,
            hidden=mlp_hidden,
            depth=mlp_depth,
            dropout=mlp_dropout,
        ).to(dev)
    else:
        net = AffineProjector(d_in, d_out).to(dev)

    # initialize network weights from the closed-form ridge solution
    net.init_from_W(W)

    # 6) dataloader over the (scaled) train pairs
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr_s),
        torch.from_numpy(Y_tr_s),
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    mse = nn.MSELoss()
    cos = nn.CosineEmbeddingLoss()

    # 7) evaluation helper:
    #    compute full-gallery MRR on the validation captions.
    #
    #    note: retrieval is always done after l2-normalization, so dot product coincides with cosine similarity.

    def eval_mrr() -> float:
        X_val_s = sanitize(apply_scaler(X_val_raw, sx.mean_, sx.scale_))
        Y_all_s = sanitize(apply_scaler(Y,        sy.mean_, sy.scale_))

        net.eval()
        with torch.no_grad():
            Qv = net(torch.from_numpy(X_val_s).to(dev)).cpu().numpy()

        Q = l2_normalize(Qv)
        G = l2_normalize(Y_all_s)
        return mrr_full(Q, G, gt_val, batch=eval_batch)

    # 8) initial evaluation: pure ridge (no gradient updates yet)

    base_mrr = eval_mrr()
    print(f"[affine init] MRR = {base_mrr:.4f}")

    # 9) fine-tuning:
    #    keep the map small and stable by mixing:
    #      - MSE loss (coordinate-wise agreement in image space)
    #      - cosine loss (directional alignment, as in retrieval settings)
    #
    #    we monitor MRR on the full gallery and keep the best checkpoint.

    best = base_mrr
    for ep in range(1, epochs + 1):
        net.train()
        loss_ep = 0.0

        for xb, yb in dl:
            xb = xb.to(dev, dtype=torch.float32, non_blocking=True)
            yb = yb.to(dev, dtype=torch.float32, non_blocking=True)

            yhat = net(xb)
            target = torch.ones(xb.size(0), device=dev)

            loss = 0.5 * mse(yhat, yb) + 0.5 * cos(yhat, yb, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            loss_ep += loss.item() * xb.size(0)

        loss_ep /= len(ds)
        mrr = eval_mrr()

        print(f"[{head_type} ep {ep:02d}] loss={loss_ep:.4f}  MRR={mrr:.4f}")
        if mrr > best:
            best = mrr
            torch.save(net.state_dict(), out_dir / f"{head_type}_projector.pt")
            np.savez(
                out_dir / "affine_scalers.npz",
                sx_mean=sx.mean_.astype("float32"),
                sx_scale=sx.scale_.astype("float32"),
                sy_mean=sy.mean_.astype("float32"),
                sy_scale=sy.scale_.astype("float32"),
            )
            print(f"  ↳ new best MRR={best:.4f}  (saved)")

    # 10) export artifacts for the next steps:
    #     - affine_val_queries.npy: projected val captions
    #     - affine_gallery.npy:     scaled image embeddings
    #     - val_gt.npy:             ground-truth indices
    #     - W_ridge.npy / bias_zero.npy: for test-time projection (step3)

    net.eval()
    with torch.no_grad():
        X_val_s = sanitize(apply_scaler(X_val_raw, sx.mean_, sx.scale_))
        Y_all_s = sanitize(apply_scaler(Y,        sy.mean_, sy.scale_))
        Qv = net(torch.from_numpy(X_val_s).to(dev)).cpu().numpy()

    np.save(out_dir / "affine_val_queries.npy", Qv.astype("float32", copy=False))
    np.save(out_dir / "affine_gallery.npy",     Y_all_s.astype("float32", copy=False))
    np.save(out_dir / "val_gt.npy",             gt_val.astype("int64",   copy=False))
    np.save(out_dir / "W_ridge.npy",            W.astype("float32"))
    np.save(out_dir / "bias_zero.npy",          np.zeros((W.shape[1],), dtype="float32"))

    print(f"saved artifacts → {out_dir}")
    print(f"[{head_type}] best MRR = {best:.4f}")

if __name__ == "__main__":
    main()
