import numpy as np


def score_variance(Z):  # Z: (M, d_hid)
    return Z.var(axis=0)

def score_mean_abs(Z):
    return np.mean(np.abs(Z), axis=0)

def score_entropy(Z, bins=50, eps=1e-12):
    # histogram entropy per dimension
    d = Z.shape[1]
    out = np.zeros(d, dtype=np.float64)
    for j in range(d):
        hist, _ = np.histogram(Z[:, j], bins=bins, density=True)
        p = hist / (hist.sum() + eps)
        out[j] = -(p * np.log(p + eps)).sum()
    return out

def topk(idx_scores, k):
    return np.argsort(idx_scores)[-k:][::-1]


def select_topk_global(Z_all, k, metric="variance"):
    if metric == "variance":
        s = score_variance(Z_all)
    elif metric == "mean_abs":
        s = score_mean_abs(Z_all)
    elif metric == "entropy":
        s = score_entropy(Z_all)
    else:
        raise ValueError(metric)
    return topk(s, k), s


def select_topk_time_local(Z_by_t, k, metric="variance"):
    # Z_by_t: list of Z_t where each is (N_t, d_hid) for a single snapshot/time
    Ks = []
    for Zt in Z_by_t:
        if metric == "variance":
            s = score_variance(Zt)
        elif metric == "mean_abs":
            s = score_mean_abs(Zt)
        elif metric == "entropy":
            s = score_entropy(Zt)
        else:
            raise ValueError(metric)
        Ks.append(topk(s, k))
    return Ks


def jaccard(a, b):
    a = set(map(int, a)); b = set(map(int, b))
    return len(a & b) / max(1, len(a | b))