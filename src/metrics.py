# src/metrics.py
# standard retrieval metrics over the full gallery (batch-friendly and numerically stable).
# implemented purely in numpy so they can be used from both pytorch and non-pytorch code.

from __future__ import annotations
import numpy as np

def _sanitize(a: np.ndarray) -> np.ndarray:
    """
    replace nan/inf values and enforce float32.
    this avoids exploding scores when computing large dot-product matrices.
    """
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)

def mrr_full(
    Q: np.ndarray,
    G: np.ndarray,
    gt: np.ndarray,
    batch: int = 2048,
) -> float:
    """
    mean reciprocal rank (mrr) against the full gallery.

    Q:  [nq, d]  query embeddings
    G:  [ng, d]  gallery embeddings
    gt: [nq]     ground-truth index in G for each query
    batch:       batch size for the matrix product

    implementation details:
      - uses float64 for the dot products for numerical stability
      - avoids a full argsort: the rank is computed as
        1 + #(elements with score strictly greater than the gt score)
    """
    Q = _sanitize(Q)
    G = _sanitize(G)
    Nq = Q.shape[0]
    rr_sum = 0.0

    # promote to float64 for more stable dot products on large galleries
    Q64 = Q.astype(np.float64, copy=False)
    G64 = G.astype(np.float64, copy=False)

    for s in range(0, Nq, batch):
        e = min(s + batch, Nq)
        sims = Q64[s:e] @ G64.T                       # [b, ng]
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

        gt_batch = gt[s:e].astype(np.int64, copy=False)
        # score of the ground-truth item for each query in the batch
        gt_scores = sims[np.arange(e - s), gt_batch]  # [b]

        # dense rank: 1 + number of gallery items with score strictly larger than the gt score
        ranks = 1 + (sims > gt_scores[:, None]).sum(axis=1)
        rr_sum += (1.0 / ranks).sum()

    return float(rr_sum) / float(Nq)


def recall_at_k_full(
    Q: np.ndarray,
    G: np.ndarray,
    gt: np.ndarray,
    k: int = 10,
    batch: int = 2048,
) -> float:
    """
    recall@k over the full gallery.

    returns the fraction of queries for which the ground-truth index
    appears among the top-k neighbors (under dot-product / cosine).
    """
    assert k >= 1
    Q = _sanitize(Q)
    G = _sanitize(G)
    Nq = Q.shape[0]
    hits = 0

    Q64 = Q.astype(np.float64, copy=False)
    G64 = G.astype(np.float64, copy=False)

    for s in range(0, Nq, batch):
        e = min(s + batch, Nq)
        sims = Q64[s:e] @ G64.T                       # [b, ng]
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

        # top-k via argpartition: partial selection is cheaper than a full sort
        topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # [b, k]
        gt_batch = gt[s:e][:, None]                           # [b, 1]
        hits += (topk_idx == gt_batch).any(axis=1).sum()

    return float(hits) / float(Nq)


def mrr_from_ranking(
    ranking: np.ndarray,
    gt: np.ndarray,
) -> float:
    """
    mrr computed from a precomputed ranking matrix.

    ranking: [nq, k]  each row contains the top-k gallery indices (sorted by similarity)
    gt:      [nq]     ground-truth gallery index for each query

    if the gt index does not appear in the top-k, the contribution is 0 for that query.
    """
    Nq, K = ranking.shape
    rr = 0.0
    for i in range(Nq):
        where = np.where(ranking[i] == gt[i])[0]
        if where.size > 0:
            rr += 1.0 / (int(where[0]) + 1)
    return rr / float(Nq)


def recall_at_k_from_ranking(
    ranking: np.ndarray,
    gt: np.ndarray,
    k: int = 10,
) -> float:
    """
    recall@k computed from a precomputed ranking matrix.

    ranking: [nq, kmax]  top-kmax gallery indices (sorted)
    gt:      [nq]        ground-truth gallery index for each query
    k:       cutoff for recall (uses ranking[:, :k])
    """
    Nq = ranking.shape[0]
    hits = (ranking[:, :k] == gt[:, None]).any(axis=1).sum()
    return float(hits) / float(Nq)
