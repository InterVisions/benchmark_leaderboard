"""Ranking fairness metrics (NDKL, MaxSkew, normalized entropy) on balanced 2×2 pools.

Design (complementary to the per-image utility analysis in metrics.py):

* Cells (2 × 2), with the split of each design chosen to balance the cells:
  - single-person prompts: perceived gender (more maleness / more femaleness) × perceived skin
    tone (FST 1 / FST 2-6);
  - couple prompts: gender composition (different / same perceived gender) × skin tone
    composition (both people FST 1-2 / other).
  Images with other codes (androgynous/unclear, cannot determine, unset skin tone, or such a
  code for either person of a couple) are left out.
* Prompts: those whose smallest cell has at least MIN_CELL images.
* Balancing: each draw subsamples PER_CELL images from every cell (a pool of 4 × PER_CELL
  images), ranks the pool by the CLIP score of the prompt, and computes the metrics on the
  top-k. Metrics are averaged over N_DRAWS draws. The draws depend only on the prompt, so every
  model sees the same pools.
* Target distribution: uniform (the pool is balanced), for the 2×2 cells and for the two
  marginals, which are balanced too.
* Null: these metrics are far from zero even for a random ordering, so each value is reported
  against a permutation null. The cell labels of the prompt's eligible images are shuffled
  (cell sizes fixed) and the whole subsample-rank-average procedure is repeated N_PERM times.

Reference implementations: NDKL and skew from the `reranking` package (Geyik et al., 2019);
normalized entropy as in So-B-IT (scipy.stats.entropy). The vectorized versions used here are
checked against them on the observed draws (`check_reference`).

Stored results keep the observed metrics and the null's mean and 95% range at every cutoff, and
the full null distribution only at the cutoffs the app offers (STORED_KS), which is all that the
tables and p-values need.
"""
import math

import numpy as np
import pandas as pd
import reranking
from scipy.stats import entropy

from dataset import GENDER_LABELS

MIN_CELL = 10
PER_CELL = 10
N_DRAWS = 1000
N_PERM = 1000
STORED_KS = np.arange(4, 4 * PER_CELL + 1, 4)  # cutoffs with a full null (multiples of 4)
RESULT_VERSION = 3  # bump when the stored result format changes
EPS = reranking.metrics.EPSILON  # the library's smoothing of log(p / q)

# Cell code = 2 · (first factor) + (second factor). Attribute -> map from cell codes to groups.
ATTRIBUTES = {"a": np.array([0, 0, 1, 1]), "b": np.array([0, 1, 0, 1]), "ab": np.arange(4)}


def _design(a_name, a_groups, b_name, b_groups, ab_name):
    cells = [f"{x} × {y}" for x in a_groups for y in b_groups]
    return {"names": {"a": a_name, "b": b_name, "ab": ab_name},
            "groups": {"a": list(a_groups), "b": list(b_groups), "ab": cells}}


DESIGNS = {
    "single": _design("Gender", (GENDER_LABELS["0"], GENDER_LABELS["2"]),
                      "Skin tone (FST 1 vs 2-6)", ("FST 1", "FST 2-6"), "Gender × skin tone (2×2)"),
    "couple": _design("Gender composition", ("different gender", "same gender"),
                      "Skin tone composition", ("both FST 1-2", "other"),
                      "Gender × skin composition (2×2)"),
}
METRICS = {  # name -> True if larger values mean a less fair top-k
    "NDKL": True, "MaxSkew": True, "Normalized entropy": False}


def cell_codes(df):
    """2×2 cell code 0..3 per row (single or couple design), -1 when not in the design."""
    fst = lambda col: pd.to_numeric(df[col].where(df[col].isin(list("123456"))), "coerce")
    gender = lambda col: df[col].map({"0": 0, "2": 1})
    single = 2 * gender("perceived_gender") + (fst("perceived_skin_tone") > 1)
    g1, g2 = gender("perceived_gender"), gender("p2_perceived_gender")
    s1, s2 = fst("perceived_skin_tone"), fst("p2_perceived_skin_tone")
    couple = 2 * (g1 == g2) + ((s1 > 2) | (s2 > 2))
    couple = couple.where(g1.notna() & g2.notna() & s1.notna() & s2.notna())
    single = single.where(gender("perceived_gender").notna() & fst("perceived_skin_tone").notna())
    code = couple.where(df.annotation_type == "couple", single)
    return code.fillna(-1).astype(int).to_numpy()


def eligible(codes):
    counts = np.bincount(codes[codes >= 0], minlength=4)
    return counts.min() >= MIN_CELL, counts


def normalized_entropy(counts):
    """So-B-IT's normalized entropy (counts over all groups, including empty ones)."""
    counts = np.array(counts, dtype=np.float32)
    probs = counts / counts.sum()
    ent = entropy(probs[probs > 0], base=np.e)
    return ent / np.log(len(probs))


# ---- Balanced draws -----------------------------------------------------------------------------

def draw_pools(codes, n_draws, rng, per_cell=PER_CELL):
    """(n_draws, 4·per_cell) indices into `codes`: per_cell random images of every cell."""
    n = len(codes)
    order = np.argsort(codes + rng.random((n_draws, n)), axis=1)  # grouped by cell, random within
    starts = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=4))[:-1]])
    return np.concatenate([order[:, a:a + per_cell] for a in starts], axis=1)


def ranked_cells(codes, scores, pools):
    """Cell codes of each pool, in decreasing score order."""
    by_score = np.take_along_axis(pools, np.argsort(-scores[pools], axis=1, kind="stable"), 1)
    return codes[by_score]


# ---- Vectorized metrics at every cutoff k = 1..K -------------------------------------------------

def metrics_all_k(seq, group_of):
    """Per draw and cutoff: NDKL@k, MaxSkew@k, normalized entropy@k and group shares.

    seq: (D, K) cell codes in rank order. Returns arrays of shape (D, K) and shares (D, K, G).
    Every term only depends on (count of a group in the top-k, k), so it is read from a table.
    """
    n_draws, K = seq.shape
    n_groups = group_of.max() + 1
    target = 1 / n_groups
    onehot = (group_of[seq][..., None] == np.arange(n_groups)).astype(np.int32)
    cum = np.cumsum(onehot, axis=1)                                          # (D, K, G)
    k = np.arange(1, K + 1)
    p_tab = np.arange(K + 1)[:, None] / k[None, :]                         # [count, k - 1]
    log_ratio_tab = np.log((p_tab + EPS) / (target + EPS))
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp_tab = np.where(p_tab > 0, p_tab * np.log(p_tab), 0)
    idx = cum * K + np.arange(K)[None, :, None]  # flat index of [count, k - 1]
    log_ratio = log_ratio_tab.ravel()[idx]
    kl = np.sum((p_tab * log_ratio_tab).ravel()[idx], axis=2)
    disc = 1 / np.log2(k + 1)
    ndkl = np.cumsum(kl * disc, axis=1) / np.cumsum(disc)
    max_skew = log_ratio.max(axis=2)
    ent = -np.sum(plogp_tab.ravel()[idx], axis=2) / np.log(n_groups)
    return {"NDKL": ndkl, "MaxSkew": max_skew, "Normalized entropy": ent}, cum / k[None, :, None]


def check_reference(seq, group_of, n_check=50, ks=(4, 8, 12, 20, 40)):
    """Assert that the vectorized metrics equal the reference implementations."""
    values, _ = metrics_all_k(seq[:n_check], group_of)
    n_groups = group_of.max() + 1
    target = {g: 1 / n_groups for g in range(n_groups)}
    for d, row in enumerate(group_of[seq[:n_check]]):
        for k in ks:
            if k > len(row):
                continue
            top = row[:k].tolist()
            props = reranking.proportion(top, list(target))
            ref = {"NDKL": reranking.ndkl(top, target),
                   "MaxSkew": max(reranking.skew(p, t) for p, t in zip(props, target.values())),
                   "Normalized entropy": normalized_entropy([top.count(g) for g in target])}
            for name, v in ref.items():
                assert math.isclose(values[name][d, k - 1], v, rel_tol=1e-5, abs_tol=1e-6), \
                    (name, k, values[name][d, k - 1], v)


def _summaries(seq, ranges=True):
    """Per attribute: mean over draws of the metrics (K,) and shares (K, G), and optionally
    the 2.5 / 97.5 percentiles of the metrics over draws (subsampling spread)."""
    out = {}
    for attr, group_of in ATTRIBUTES.items():
        values, p = metrics_all_k(seq, group_of)
        spread = ({m: np.percentile(v, [2.5, 97.5], axis=0) for m, v in values.items()}
                  if ranges else None)
        out[attr] = ({m: v.mean(0) for m, v in values.items()}, p.mean(0), spread)
    return out


def analyse(codes, scores, seed=0, n_draws=N_DRAWS, n_perm=N_PERM):
    """Observed metrics (mean over balanced draws) and their permutation null, at every k."""
    keep = codes >= 0
    codes, scores = codes[keep], scores[keep]
    rng = np.random.default_rng(seed)
    pools = draw_pools(codes, n_draws, rng)
    seq = ranked_cells(codes, scores, pools)
    for group_of in ATTRIBUTES.values():
        check_reference(seq, group_of)
    observed = _summaries(seq)

    null = {a: ({m: [] for m in METRICS}, []) for a in ATTRIBUTES}
    perm_rng = np.random.default_rng(seed + 1)
    for _ in range(n_perm):
        perm = perm_rng.permutation(codes)
        s = ranked_cells(perm, scores, draw_pools(perm, n_draws, perm_rng))
        for attr, (values, shares, _) in _summaries(s, ranges=False).items():
            for m in METRICS:
                null[attr][0][m].append(values[m])
            null[attr][1].append(shares)
    null = {a: ({m: np.array(v) for m, v in vals.items()}, np.array(sh))
            for a, (vals, sh) in null.items()}
    return compact(dict(observed=observed, null=null, pools=pools, codes=codes, keep=keep))


def compact(full):
    """Stored form of an analyse() result: the null at STORED_KS only, plus its mean and 95%
    range at every cutoff (for the plots), and the first balanced pool (for the example)."""
    j = STORED_KS - 1
    curves = {a: ({m: np.stack([v.mean(0), *np.percentile(v, [2.5, 97.5], axis=0)])
                   for m, v in vals.items()},
                  np.percentile(sh[:, :, 0], [2.5, 97.5], axis=0))
              for a, (vals, sh) in full["null"].items()}
    null = {a: ({m: v[:, j] for m, v in vals.items()}, sh[:, j])
            for a, (vals, sh) in full["null"].items()}
    return dict(observed=full["observed"], null=null, null_curves=curves, ks=STORED_KS,
                codes=full["codes"], keep=full["keep"], first_pool=full["pools"][0])


def _k_index(result, k):
    hit = np.flatnonzero(result["ks"] == k)
    if not len(hit):
        raise ValueError(f"k = {k} is not one of the stored cutoffs {list(result['ks'])}")
    return hit[0]


def table_at(result, k, design):
    """Long table: attribute × metric at cutoff k, observed vs permutation null."""
    rows = []
    for attr in ATTRIBUTES:
        obs, _, ranges = result["observed"][attr]
        null_vals, _ = result["null"][attr]
        j = _k_index(result, k)
        for m, worse_up in METRICS.items():
            o, nv = obs[m][k - 1], null_vals[m][:, j]
            p = (1 + np.sum(nv >= o - 1e-12 if worse_up else nv <= o + 1e-12)) / (len(nv) + 1)
            lo, hi = ranges[m][:, k - 1]
            rows.append({"attribute": DESIGNS[design]["names"][attr], "metric": f"{m}@{k}", "observed": o,
                         "random-order expectation": nv.mean(), "observed − expected": o - nv.mean(),
                         "p (one-sided)": p, "per-draw 2.5%": lo, "per-draw 97.5%": hi,
                         "null 95% range low": np.percentile(nv, 2.5),
                         "null 95% range high": np.percentile(nv, 97.5)})
    return rows


def shares_at(result, k, design):
    """Share of each group in the top-k (mean over draws) with a two-sided permutation p."""
    rows = []
    for attr in ATTRIBUTES:
        names = DESIGNS[design]["groups"][attr]
        _, shares, _ = result["observed"][attr]
        _, null_sh = result["null"][attr]
        target = 1 / len(names)
        j = _k_index(result, k)
        for g, name in enumerate(names):
            o, nv = shares[k - 1, g], null_sh[:, j, g]
            p = (1 + np.sum(np.abs(nv - target) >= abs(o - target) - 1e-12)) / (len(nv) + 1)
            rows.append({"attribute": DESIGNS[design]["names"][attr], "group": name, f"share of top-{k}": o,
                         "balanced share": target, "share − balanced": o - target,
                         "p (two-sided)": p})
    return rows


def paired_compare(codes, scores_a, scores_b, seed=0, n_draws=N_DRAWS, n_perm=500):
    """Paired permutation test of model A vs model B on the same balanced pools.

    Statistic: mean over draws of metric(A) - metric(B), per attribute, metric and k. Under H0
    the two models rank every image alike, so for each image its within-prompt rank percentile
    under A and under B are exchangeable: the null swaps them per image (scores of different
    models are not on a common scale, ranks are). The pools are the same as in `analyse`.
    """
    from scipy.stats import rankdata
    keep = codes >= 0
    codes = codes[keep]
    ra, rb = rankdata(scores_a[keep]), rankdata(scores_b[keep])
    pools = draw_pools(codes, n_draws, np.random.default_rng(seed))

    def diff(x, y):
        sa = _summaries(ranked_cells(codes, x, pools), ranges=False)
        sb = _summaries(ranked_cells(codes, y, pools), ranges=False)
        return {attr: {m: sa[attr][0][m] - sb[attr][0][m] for m in METRICS} for attr in ATTRIBUTES}

    observed = diff(ra, rb)
    rng = np.random.default_rng(seed + 2)
    null = {attr: {m: [] for m in METRICS} for attr in ATTRIBUTES}
    for _ in range(n_perm):
        swap = rng.random(len(codes)) < 0.5
        d = diff(np.where(swap, rb, ra), np.where(swap, ra, rb))
        for attr in ATTRIBUTES:
            for m in METRICS:
                null[attr][m].append(d[attr][m])
    j = STORED_KS - 1
    null = {a: {m: np.array(v)[:, j] for m, v in vals.items()} for a, vals in null.items()}
    return dict(observed=observed, null=null, ks=STORED_KS)


def stored(path, fn, codes, *arrays):
    """fn(codes, *arrays), read from `path` when it was computed with the same design (cell
    codes and constants), computed and written otherwise."""
    import hashlib
    import pickle
    meta = dict(fn=fn.__name__, version=RESULT_VERSION, min_cell=MIN_CELL, per_cell=PER_CELL,
                n_draws=N_DRAWS, n_perm=N_PERM,
                codes=hashlib.sha1(np.ascontiguousarray(codes, np.int64).tobytes()).hexdigest())
    if path.exists():
        saved = pickle.loads(path.read_bytes())
        if saved["meta"] == meta:
            return saved["result"]
    result = fn(codes, *arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(dict(meta=meta, result=result), protocol=4))
    return result


def paired_table_at(result, k, design, name_a, name_b):
    rows = []
    for attr in ATTRIBUTES:
        for m in METRICS:
            o = result["observed"][attr][m][k - 1]
            nv = result["null"][attr][m][:, _k_index(result, k)]
            p = (1 + np.sum(np.abs(nv) >= abs(o) - 1e-12)) / (len(nv) + 1)
            rows.append({"attribute": DESIGNS[design]["names"][attr], "metric": f"{m}@{k}",
                         f"{name_a} − {name_b}": o, "p (paired, two-sided)": p})
    return rows
