# src/step3_test.py

# step 3: creation of the final submission file.
#
# goal:
#   take the embeddings from the official test set (text side),
#   apply the cross-modal projection learned in step 2,
#   and export the resulting image-space vectors in the csv format expected
#   by the leaderboard.
#
# background:
#   the challenge evaluates how well text and image embeddings are aligned.
#   in step 2 we learned a map f : x_text → y_image, either linear
#   (closed-form ridge) or linear + small residual mlp. this script simply
#   applies that map to unseen test data.
#
# export modes:
#   1) affine mode:
#      uses the closed-form ridge solution y = x_s @ w + b, i.e. a pure
#      linear alignment in the spirit of classic cca / procrustes mappings.
#   2) model forward mode:
#      loads the fine-tuned projector checkpoint (.pt) from step 2 and runs
#      a forward pass (optionally with a shallow residual mlp on top).
#
# output:
#   csv with columns:
#       id, embedding
#   where embedding is a json array of floats (one vector per test caption).

import os, sys, csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# cuda quality-of-life flags (avoid fragmentation and enable fast kernels)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# import cfg from project root
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))
from config import CFG
from utils import apply_scaler


# projector modules (minimal versions of the step-2 architectures)

class AffineProjector(nn.Module):
    """plain affine map y = w x + b, used when there is no residual head."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class ResidualMLPProjectorPreNorm(nn.Module):
    """
    prenorm residual projector:
        y0 = base(x)
        y  = y0 + mlp( ln(y0) )
    this mirrors the small adapter-style mlps used in many recent
    multimodal models (ln → ff → nonlinearity → ff → residual add).
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 1024, layers: int = 2):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)

        blocks = []
        if layers <= 1:
            blocks.append(nn.Linear(d_out, d_out))
        else:
            blocks.append(nn.Linear(d_out, hidden))
            blocks.append(nn.GELU())
            for _ in range(layers - 2):
                blocks.append(nn.Linear(hidden, hidden))
                blocks.append(nn.GELU())
            blocks.append(nn.Linear(hidden, d_out))
        self.mlp = nn.Sequential(*blocks)

        # zero-init last linear layer so that training starts from the pure
        # affine solution and gradually learns non-linear corrections.
        last = self.mlp[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        return y0 + self.mlp(self.norm(y0))


class ResidualMLPProjectorLegacy(nn.Module):
    """
    legacy residual projector without layernorm:
        y0 = base(x)
        y  = y0 + res(y0)
    kept for compatibility with older checkpoints that did not use ln.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 1024, layers: int = 2):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)

        blocks = []
        if layers <= 1:
            blocks.append(nn.Linear(d_out, d_out))
        else:
            blocks.append(nn.Linear(d_out, hidden))
            blocks.append(nn.GELU())
            for _ in range(layers - 2):
                blocks.append(nn.Linear(hidden, hidden))
                blocks.append(nn.GELU())
            blocks.append(nn.Linear(hidden, d_out))
        self.res = nn.Sequential(*blocks)

        last = self.res[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        return y0 + self.res(y0)


class ResidualResSeqPreNorm(nn.Module):
    """
    residual projector used in the current step-2 implementation:
        y0   = base(x)
        yhat = y0 + res(y0)
    where `res` is a sequential block:
        [ln, linear, gelu, dropout, linear] × depth
    the last linear layer is zero-initialized so that the network starts
    as a pure affine map and only later deviates from it.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int = 1024,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        nn.init.zeros_(self.base.bias)

        blocks = []
        d = d_out
        for _ in range(layers):
            blocks.extend(
                [
                    nn.LayerNorm(d),
                    nn.Linear(d, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d),
                ]
            )
        self.res = nn.Sequential(*blocks)

        # zero-init only the last linear layer
        for m in reversed(self.res):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.base(x)
        return y0 + self.res(y0)


# ---------------------------------------------------------------------
# small local helpers
# ---------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_npz_matrix(npz_path: str) -> np.ndarray:
    """
    load the main matrix from the test npz.
    we scan a few common keys because train/test files may use slightly
    different naming conventions.
    """
    data = np.load(npz_path)
    for k in ("X", "X_text", "queries", "txt", "arr_0"):
        if k in data and data[k].ndim == 2:
            return data[k]
    for k in data.files:
        if data[k].ndim == 2:
            return data[k]
    raise ValueError(f"no 2d matrix found in {npz_path}")


def _load_test_ids(csv_path: str):
    """
    load official test ids from the captions file.
    the leaderboard script uses these ids to match predictions and ground
    truth, so we preserve the original ordering.
    """
    ids = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header and header[0].strip().lower() != "id":
            # if the first column is not "id", rewind and treat as header-less
            f.seek(0)
            r = csv.reader(f)
        for row in r:
            if not row:
                continue
            try:
                ids.append(int(row[0]))
            except Exception:
                ids.append(row[0])
    return ids


def _tensor_shape(state, key):
    t = state.get(key, None)
    if t is None:
        return None
    return tuple(t.shape)


def _detect_variant_from_state(state_dict: dict, head_hint: str) -> str:
    """
    lightweight heuristic to detect which projector variant was used
    in step 2 by inspecting checkpoint keys.

    returns one of:
        'affine'          → plain linear map
        'resmlp_prenorm'  → ln + mlp blocks (norm./mlp. keys)
        'resseq_prenorm'  → sequential residual with ln as res.0
        'resmlp_legacy'   → residual mlp without explicit layernorm
    """
    keys = list(state_dict.keys())

    # if the config explicitly says "affine", trust that first
    if head_hint == "affine":
        return "affine"

    # explicit norm/mlp modules → prenorm projector
    if any(k.startswith("norm.") or k.startswith("mlp.") for k in keys):
        return "resmlp_prenorm"

    # generic residual blocks under "res."
    if any(k.startswith("res.") for k in keys):
        wshape = _tensor_shape(state_dict, "res.0.weight")
        # layernorm parameters are 1d; linear weights are 2d
        if wshape is not None and len(wshape) == 1:
            return "resseq_prenorm"
        return "resmlp_legacy"

    # fallback to affine if nothing else matches
    return "affine"


# ---------------------------------------------------------------------
# main entrypoint
# ---------------------------------------------------------------------

def main():
    cfg = CFG.step3

    # log mode and chosen run
    mode = (
        "AFFINE_WB"
        if getattr(cfg, "USE_AFFINE_WB", False)
        else ("MODEL_PT" if getattr(cfg, "USE_MODEL_PT", True) else "??")
    )
    print(f"[step3] mode: {mode}")
    run_dir = Path(cfg.SCALERS_NPZ).parent if hasattr(cfg, "SCALERS_NPZ") else Path(".")
    print(f"[step3] run dir: {run_dir}")

    out_dir = Path(cfg.OUT_DIR)
    _ensure_dir(out_dir)
    out_csv = out_dir / cfg.CSV_NAME

    # 1) load test embeddings and ids
    x_test = _load_npz_matrix(cfg.TEST_NPZ).astype(np.float32, copy=False)
    ids = _load_test_ids(cfg.TEST_IDS_CSV)
    if len(ids) != x_test.shape[0]:
        print(
            f"[warn] mismatch between ids ({len(ids)}) and embeddings "
            f"({x_test.shape[0]}) → falling back to 1..n indexing."
        )
        ids = list(range(1, x_test.shape[0] + 1))

    # 2) load scalers from step 2 (μ/σ from training set)
    sc = np.load(cfg.SCALERS_NPZ)
    sx_mean, sx_scale = sc["sx_mean"], sc["sx_scale"]
    d_in = int(sx_mean.shape[0])
    d_out = int(sc["sy_mean"].shape[0]) if "sy_mean" in sc else None

    # 3) standardize test embeddings with the same statistics used in training
    x_s = apply_scaler(x_test, sx_mean, sx_scale).astype(np.float32, copy=False)

    # 4) apply the learned map: affine only or full model
    if getattr(cfg, "USE_AFFINE_WB", False):
        # closed-form ridge projection
        w = np.load(cfg.W_NUMPY).astype(np.float32, copy=False)
        bias = np.load(cfg.BIAS_NUMPY).astype(np.float32, copy=False)
        assert w.shape[0] == d_in
        if d_out is not None:
            assert w.shape[1] == d_out
        y_pred = x_s @ w + bias
        print(f"[step3] affine export → y_pred {y_pred.shape}")

    elif getattr(cfg, "USE_MODEL_PT", True):
        # full model: affine + optional residual projector
        device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        head_hint = str(cfg.HEAD).lower()

        state = torch.load(cfg.MODEL_PT, map_location="cpu")
        variant = _detect_variant_from_state(state, head_hint)
        print(f"[step3] loading head={head_hint} | detected_variant={variant}")

        if variant == "affine":
            model = AffineProjector(d_in, d_out)
        elif variant == "resmlp_prenorm":
            hidden = int(getattr(cfg, "MLP_HIDDEN", 1024))
            layers = int(getattr(cfg, "MLP_LAYERS", 2))
            model = ResidualMLPProjectorPreNorm(d_in, d_out, hidden=hidden, layers=layers)
        elif variant == "resseq_prenorm":
            hidden = int(getattr(cfg, "MLP_HIDDEN", 1024))
            layers = int(getattr(cfg, "MLP_LAYERS", 2))
            model = ResidualResSeqPreNorm(d_in, d_out, hidden=hidden, layers=layers)
        else:  # resmlp_legacy
            hidden = int(getattr(cfg, "MLP_HIDDEN", 1024))
            layers = int(getattr(cfg, "MLP_LAYERS", 2))
            model = ResidualMLPProjectorLegacy(d_in, d_out, hidden=hidden, layers=layers)

        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print("[step3][warn] strict=True failed, retrying with strict=False:", e)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[step3][warn] missing={missing} unexpected={unexpected}")

        model.to(device).eval()

        y_pred_chunks = []
        bs = 8192
        with torch.no_grad():
            for s in range(0, x_s.shape[0], bs):
                e = min(s + bs, x_s.shape[0])
                xb = torch.from_numpy(x_s[s:e]).to(device, dtype=torch.float32, non_blocking=True)
                y = model(xb).cpu().numpy()
                y_pred_chunks.append(y.astype(np.float32, copy=False))
        y_pred = np.concatenate(y_pred_chunks, axis=0)
        print(f"[step3] model export ({variant}) → y_pred {y_pred.shape}")

    else:
        raise ValueError("step3 config error: specify either USE_AFFINE_WB or USE_MODEL_PT")

    # 5) write csv in the expected format
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "embedding"])
        for i, row in zip(ids, y_pred):
            w.writerow([i, json.dumps(row.tolist(), ensure_ascii=False)])
    print(f"[step3] written: {out_csv}  ({y_pred.shape[0]} rows, dim={y_pred.shape[1]})")


if __name__ == "__main__":
    main()
