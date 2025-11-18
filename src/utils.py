import numpy as np
from sklearn.preprocessing import StandardScaler

def ensure_f32(a: np.ndarray) -> np.ndarray:
    a = a.astype("float32", copy=False)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def clip_std(a: np.ndarray, s: float = 8.0) -> np.ndarray:
    # clip after standard-scaler to avoid numeric outliers
    np.clip(a, -s, s, out=a)
    return a

def l2_normalize(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (a / n).astype("float32")

def load_train_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = ensure_f32(d["captions/embeddings"])
    Y = ensure_f32(d["images/embeddings"])
    lab = d["captions/label"]
    if lab.ndim == 2:  # some packs store one-hot; reduce to argmax
        lab = np.argmax(lab, axis=1)
    lab = lab.astype("int64")
    return X, Y, lab

def train_val_split(n, val_frac=0.1, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    if val_frac is None:
        val_frac = 0.1
    try:
        vf = float(val_frac)
    except Exception:
        vf = 0.1

    if vf >= 1:
        n_val = int(vf)
    elif vf <= 0:
        n_val = max(1, int(round(0.1 * n)))   # fallback 10%
    else:
        n_val = max(1, int(round(n * vf)))

    n_val = min(n_val, n - 1)                 # tieni almeno 1 in train
    return idx[n_val:], idx[:n_val]


def fit_scalers(X_tr: np.ndarray, Y_tr: np.ndarray):
    sx = StandardScaler()
    sy = StandardScaler()
    X_tr_s = clip_std(ensure_f32(sx.fit_transform(X_tr)))
    Y_tr_s = clip_std(ensure_f32(sy.fit_transform(Y_tr)))
    return sx, sy, X_tr_s, Y_tr_s

def apply_scaler(A: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.where(scale == 0, 1.0, scale)
    return ((A - mean) / scale_safe).astype("float32")

def sanitize_finite(a):
    # force finite, then re-normalize (important if rows collapsed)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
    # if any row got zeroed, l2 will handle it
    return a

def debug_finite(name, a):
    n = a.shape[0]
    bad = (~np.isfinite(a)).reshape(n, -1).any(axis=1).sum()
    mx = np.max(np.abs(a[np.isfinite(a)])) if np.isfinite(a).any() else np.inf
    print(f"[check] {name}: rows_with_nonfinite={bad}/{n}, max_abs={mx:.3e}")

