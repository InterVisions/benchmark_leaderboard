"""Per-image retrieval utility and per-group summaries.

Two per-image utilities for a prompt q and one of its relevant images i with score s_i:

* In-dataset AUC: fraction of negatives (InterVisions images collected for
  other prompts) scored below s_i, ties counted 1/2. Other relevant images of q are not
  negatives, so groups do not compete with each other. Chance level is 0.5 for any group size.
* Distractor rank (fixed external gallery, e.g. COCO): rank of image i when inserted alone
  into the distractor set, rank = 1 + #distractors scored above s_i. Its normalized form
  1 - (rank - 1) / N is the AUC against the distractors. R@k = fraction of images with rank <= k,
  chance k / (N + 1) for any group size.

Groups are summarized by medians (robust to the few images CLIP misses badly); means are kept
for reference.
"""
from math import comb

import numpy as np
import pandas as pd


def auc_vs(pos_scores, neg_scores):
    neg = np.sort(neg_scores)
    below = np.searchsorted(neg, pos_scores, side="left")
    ties = np.searchsorted(neg, pos_scores, side="right") - below
    return (below + 0.5 * ties) / len(neg)


def rank_vs(pos_scores, distractor_scores):
    d = np.sort(distractor_scores)
    return 1 + len(d) - np.searchsorted(d, pos_scores, side="right")


def median_ci(x, level=0.95):
    """Distribution-free CI of the median from order statistics (binomial / sign-test argument).

    The interval (x_(j), x_(n-j+1)) covers the median with probability 1 - 2 P(Bin(n, 1/2) < j).
    No interval reaches 95% for n <= 5; for n = 6..8 it is (min, max); from n = 9 it excludes
    the extremes. Returns (nan, nan) when no interval reaches `level`.
    """
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cdf = np.cumsum([comb(n, i) for i in range(n + 1)]) / 2 ** n  # P(Bin <= i)
    ok = [j for j in range(1, n // 2 + 1) if 1 - 2 * cdf[j - 1] >= level]
    if not ok:
        return np.nan, np.nan
    j = max(ok)
    return x[j - 1], x[n - j]


def permutations(n, n_perm=5000, seed=0):
    """Random permutations of range(n), shared by all groups of one prompt."""
    rng = np.random.default_rng(seed)
    return rng.permuted(np.tile(np.arange(n), (n_perm, 1)), axis=1)


def perm_test_median(values, in_group, perms):
    """Median(group) - median(rest) and its two-sided permutation p-value (group sizes fixed)."""
    values = np.asarray(values, float)
    n_in = int(in_group.sum())
    if n_in == 0 or n_in == len(values):
        return np.nan, np.nan
    obs = np.median(values[in_group]) - np.median(values[~in_group])
    shuffled = values[perms]
    diffs = np.median(shuffled[:, :n_in], 1) - np.median(shuffled[:, n_in:], 1)
    p = (1 + np.sum(np.abs(diffs) >= abs(obs) - 1e-12)) / (len(perms) + 1)
    return obs, p


def group_summary(labels, auc_in, rank_ext, ks=(1, 5, 10), n_perm=5000, seed=0):
    """One row per group of a prompt's relevant images. Unlabelled images ("") are dropped.

    Medians are the headline (CI from order statistics, Δ vs the other labelled images of the
    prompt with a permutation p-value); means are kept for reference.
    """
    labels = np.asarray(labels)
    keep = labels != ""
    labels, auc_in, rank_ext = labels[keep], auc_in[keep], rank_ext[keep]
    perms = permutations(len(labels), n_perm, seed)
    rows = []
    for g in sorted(set(labels), key=lambda g: (-np.sum(labels == g), g)):
        m = labels == g
        a_lo, a_hi = median_ci(auc_in[m])
        a_d, a_p = perm_test_median(auc_in, m, perms)
        r_lo, r_hi = median_ci(rank_ext[m])
        r_d, r_p = perm_test_median(rank_ext, m, perms)
        row = {"group": g, "n": int(m.sum()),
               "AUC median": np.median(auc_in[m]), "AUC CI low": a_lo, "AUC CI high": a_hi,
               "AUC Δ median": a_d, "AUC p": a_p, "AUC mean": auc_in[m].mean(),
               "rank median": np.median(rank_ext[m]), "rank CI low": r_lo, "rank CI high": r_hi,
               "rank Δ median": r_d, "rank p": r_p, "rank mean": rank_ext[m].mean()}
        for k in ks:
            row[f"R@{k}"] = float(np.mean(rank_ext[m] <= k))
        rows.append(row)
    columns = ["group", "n", "AUC median", "AUC CI low", "AUC CI high", "AUC Δ median", "AUC p",
               "AUC mean", "rank median", "rank CI low", "rank CI high", "rank Δ median",
               "rank p", "rank mean"] + [f"R@{k}" for k in ks]
    return pd.DataFrame(rows, columns=columns)


# ---- Evidence grading of single-prompt claims ---------------------------------------------------
# A claim is "model m ranks group g lower (or higher) than the other labelled images of prompt q".
# Claims are never pooled across prompts: whether a direction is harmful depends on the prompt.

ALPHA = 0.05
STRONG_N = 9   # from n = 9 the 95% CI of the median excludes the extremes
EVIDENCE_RULES = f"""**Evidence levels** for a claim *"model m ranks group g lower/higher than the rest of
prompt q"*, judged on the COCO median-rank test (median difference vs the rest of the prompt,
permutation p-value):

* **strong**: n ≥ {STRONG_N}; Holm-adjusted p < {ALPHA} (family = all assessed groups of this
  grouping in this prompt, for this model); and the in-dataset AUC median difference points the
  same way with p < {ALPHA}.
* **suggestive**: not strong, but raw p < {ALPHA} and the in-dataset AUC points the same way.
* **protocols disagree**: raw p < {ALPHA}, but the in-dataset AUC points the other way.
* **uncertain**: raw p ≥ {ALPHA}. Not evidence of no difference, only that none is detectable at this n.
* **not assessed**: n below the minimum group size M (no 95% CI for the median below n = 6).

*Other models* counts the other CLIP models where the same group goes the same way with raw
p < {ALPHA}: replication beyond the model the claim is about."""


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaNs are ignored and kept)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(order)])
    out[order] = np.minimum(adj, 1)
    return out


def grade(table, min_n=6):
    """Add direction, Holm-adjusted p and evidence level to a group_summary table."""
    t = table.copy()
    assessed = t.n >= min_n
    t["rank p Holm"] = np.nan
    t.loc[assessed, "rank p Holm"] = holm(t.loc[assessed, "rank p"])
    t["direction"] = np.select([t["rank Δ median"] > 0, t["rank Δ median"] < 0],
                               ["ranked lower", "ranked higher"], "no difference")
    agree = np.sign(t["AUC Δ median"]) == -np.sign(t["rank Δ median"])
    raw = t["rank p"] < ALPHA
    strong = (t.n >= STRONG_N) & (t["rank p Holm"] < ALPHA) & agree & (t["AUC p"] < ALPHA)
    t["evidence"] = np.select(
        [~assessed, strong, raw & agree, raw & ~agree],
        ["not assessed", "strong", "suggestive", "protocols disagree"], "uncertain")
    return t


# ---- Direction-agnostic disparity (for averaging across prompts) ------------------------------

def rank_spread(labels, rank, min_n=6, n_perm=2000, seed=0):
    """Worst-to-best spread of group median COCO ranks, in log2 units, against its null.

    spread = max_g log2(median rank_g) - min_g log2(median rank_g) over the groups with n >= min_n
    (1 = the worst group's median rank is twice the best group's). It is positive even for
    random labels, so it is compared with a permutation null (labels shuffled among the images
    of the assessed groups, group sizes fixed). Returns (spread, null mean, one-sided p, #groups);
    NaNs when fewer than two groups are assessed.
    """
    labels = np.asarray(labels)
    groups = [g for g in sorted(set(labels)) if g != "" and np.sum(labels == g) >= min_n]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan, len(groups)
    v = np.concatenate([np.log2(rank[labels == g]) for g in groups])
    bounds = np.cumsum([0] + [int(np.sum(labels == g)) for g in groups])

    def spread(x):  # x: (..., n) values ordered by group segment
        med = np.stack([np.median(x[..., a:b], -1) for a, b in zip(bounds[:-1], bounds[1:])], -1)
        return med.max(-1) - med.min(-1)
    obs = spread(v)
    null = spread(v[permutations(len(v), n_perm, seed)])
    p = (1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)
    return float(obs), float(null.mean()), float(p), len(groups)
