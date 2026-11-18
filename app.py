"""InterVisions retrieval fairness leaderboard.

Shows aggregate results of text-to-image retrieval models on the InterVisions benchmark, read from
data/ (tables of numbers exported from the private benchmark; see check_export.py). The images,
their annotations and the embeddings are not public, and nothing here is per image.

Run:  python app.py [--host 0.0.0.0] [--port 7860]

The published leaderboard is the static page index.html (static/leaderboard.js), which shows the
same views and numbers without a server; this Gradio version is for running it locally.
"""
import argparse
import base64
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage telemetry to Gradio
import gradio as gr  # noqa: E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6  # also runs on Gradio 5
sys.path.insert(0, str(ROOT / "evaluation"))

import check_export  # noqa: E402
from dataset import FST_COLORS, GENDER_COLORS, LABEL_COLORS, _mix, label_color  # noqa: E402
from metrics import EVIDENCE_RULES  # noqa: E402

check_export.main(ROOT)  # refuse to serve data that is not aggregate-only

DATA = ROOT / "data"
META = json.loads((DATA / "meta.json").read_text())
MODELS = [m["key"] for m in META["models"]]
PROMPTS = [p["prompt"] for p in META["prompts"]]
DESIGN = {p["prompt"]: p["design"] for p in META["prompts"]}
RF_PROMPTS = [p["prompt"] for p in META["prompts"] if p["ranking_fairness"]]
KS, MIN_N = META["ks"], META["min_n"]
RF_DESIGNS, RF_METRICS = META["rf_designs"], META["rf_metrics"]  # metric -> larger is less fair
RFC, DIPC = META["rf_constants"], META["dip_constants"]
LEADERBOARD = META["leaderboard"]
GROUPINGS = META["groupings"]


def _csv(name):
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False, na_values=[""])


GROUPS, SPREAD, UTILITY = _csv("groups"), _csv("spread"), _csv("utility")
RF_METRIC_T, RF_SHARES, RF_CURVES = _csv("rf_metrics"), _csv("rf_shares"), _csv("rf_curves")
RF_CELLS, PAIRED = _csv("rf_cells"), _csv("paired")
for t in (GROUPS, SPREAD, UTILITY, RF_METRIC_T, RF_SHARES, RF_CURVES, RF_CELLS, PAIRED):
    for c in t.columns:
        if t[c].dtype == object:
            t[c] = t[c].fillna("")


def _k(k):
    return int(min(KS, key=lambda v: abs(v - float(k or 12))))


def _short(model):
    return model.replace(" / ", " ")


def _sign(metric):
    """+1 / -1 so that sign × (a − b) is larger when a is less fair (entropy: higher = fairer)."""
    return 1 if RF_METRICS[metric] else -1


def _metric_label(metric, k=None):
    return f"{metric}{f'@{k}' if k else ''} {'↓' if RF_METRICS[metric] else '↑'}"


# ---- Styling --------------------------------------------------------------------------------------

def _p_shade(p):
    if not isinstance(p, (float, int)) or not np.isfinite(p) or p >= 0.05:
        return ""
    return "background-color:#D55E00;color:white" if p < 0.01 else \
        "background-color:#F6C9A6;color:black"


def _style(table, pcols=()):
    """Number formats, group cells in the annotation colors, and p-value shading."""
    if table.empty:
        return table

    def cell(label):
        c = label_color(label, default="")
        if not c:
            return ""
        dark = sum(int(c[i:i + 2], 16) for i in (1, 3, 5)) < 3 * 140
        return f"background-color:{c};color:{'white' if dark else 'black'}"
    fmt = {c: "{:g}" for c in ("COCO median rank", "COCO Δ median rank vs rest") if c in table}
    fmt.update({c: "{:.1f}" for c in ("COCO mean rank",) if c in table})
    fmt.update({c: "{:.1%}" for c in table if c.startswith(("share", "balanced share",
                                                             "Lowest group share"))})
    styler = table.style.map(cell, subset=["group"]) if "group" in table else table.style
    pcols = [c for c in pcols if c in table]
    if pcols:
        styler = styler.map(_p_shade, subset=pcols)
    return styler.format(fmt, precision=3, na_rep="")


P_LEGEND = ("**p shading**: <span style='background:#D55E00;color:white;padding:0 4px'>p &lt; 0.01"
            "</span> <span style='background:#F6C9A6;color:black;padding:0 4px'>p &lt; 0.05</span>"
            " · {} Not corrected for multiple comparisons unless stated, so an isolated "
            "p &lt; 0.05 is a hint, not a finding.")


# ---- Leaderboard ----------------------------------------------------------------------------------

LB_WARNING = f"""**Summary only.** Each number is an average over a handful of prompts and hides
which group is favoured and where; a single number is not a verdict on a model, and the columns are
not combined into a score. Fairness columns are **excess over chance** (random labels or random
order): 0 = as fair as chance. Read them next to utility: a model can look fair by being bad for
everyone. Differences between two models are tested only in *Compare two models*. Groups with
fewer than {MIN_N} images are not published and not assessed."""

LB_INFO = f"""* **COCO median rank** (median over prompts of the median rank of the relevant images among
  {DIPC['coco_images']:,} COCO val2017 distractors) and **COCO R@10** (mean over prompts): utility.
* **Excess spread** (difference in performance): per prompt, log2 of the median COCO rank of the
  worst group (n ≥ {MIN_N}) over that of the best group, minus its permutation-null expectation;
  mean over the prompts where ≥ 2 groups are assessed. +1 = the worst-to-best ratio of median ranks
  is twice what random labels give. Direction-agnostic, so reversals between prompts do not cancel.
* **Strong / suggestive claims**: single-prompt claims of this grouping (see *Evidence*), summed
  over prompts. A count, not an average; fewer claims can also mean less statistical power.
* **Ranking fairness at k** (balanced 2×2 pools, see *Ranking fairness*), mean over the analysed
  prompts of observed − random-order expectation: **NDKL** and **MaxSkew** excess, and **entropy
  deficit** (expectation − observed normalized entropy). Larger = less fair than random.
* **Lowest group share**: share of the top-k held by the least represented group (mean over
  prompts); the balanced value is 1 / #groups.
* The groups differ between the two analyses: difference in performance uses the codebook groupings
  (e.g. 3 skin-tone bins), ranking fairness the balanced 2×2 split (FST 1 / 2-6 for single-person
  prompts, both FST 1-2 / other for couples)."""


def _lb_rows(config, k):
    cfg = LEADERBOARD[config]
    design, attr, grouping = cfg["design"], cfg["attribute"], cfg["grouping"]
    terms = [t for t in PROMPTS if DESIGN[t] == design]
    rf_terms = [t for t in RF_PROMPTS if DESIGN[t] == design] if attr else []
    attr_name = RF_DESIGNS[design]["names"][attr] if attr else None
    rows = []
    for m in MODELS:
        for t in terms:
            u = UTILITY[(UTILITY.model == m) & (UTILITY.prompt == t)].iloc[0]
            s = SPREAD[(SPREAD.model == m) & (SPREAD.prompt == t) & (SPREAD.grouping == grouping)]
            ev = GROUPS[(GROUPS.model == m) & (GROUPS.prompt == t) & (GROUPS.grouping == grouping)]
            row = {"model": m, "prompt": t, "COCO median rank ↓": u["COCO median rank"],
                   "COCO R@10 ↑": u["COCO R@10"],
                   "Excess spread ↓": (s.spread - s["null mean"]).iloc[0] if len(s) else np.nan,
                   "spread p": s.p.iloc[0] if len(s) else np.nan,
                   "Strong claims": int((ev.evidence == "strong").sum()),
                   "Suggestive claims": int((ev.evidence == "suggestive").sum())}
            if t in rf_terms:
                mt = RF_METRIC_T[(RF_METRIC_T.model == m) & (RF_METRIC_T.prompt == t)
                                 & (RF_METRIC_T.k == k) & (RF_METRIC_T.attribute == attr_name)]
                for r in mt.to_dict("records"):
                    col = f"Entropy@{k} deficit ↓" if r["metric"] == "Normalized entropy" \
                        else f"{r['metric']}@{k} excess ↓"
                    row[col] = _sign(r["metric"]) * r["observed − expected"]
                    row[f"{col.split(' ')[0]} p"] = r["p (one-sided)"]
                sh = RF_SHARES[(RF_SHARES.model == m) & (RF_SHARES.prompt == t)
                               & (RF_SHARES.k == k) & (RF_SHARES.attribute == attr_name)]
                row[f"Lowest group share@{k} ↑ (balanced {1 / len(sh):.0%})"] = sh.share.min()
            rows.append(row)
    return pd.DataFrame(rows), terms, rf_terms, attr_name


def _lb_html(main, k):
    """Summary table as HTML: column groups by analysis, best value per column in bold."""
    groups = [("", "model")]
    for c in main.columns[1:]:
        block = ("Utility" if c.startswith("COCO") else
                 f"Ranking fairness (top-{k})" if f"@{k}" in c else "Difference in performance")
        groups.append((block, c.replace(f"@{k}", "")))
    best = pd.DataFrame(False, index=main.index, columns=main.columns)
    for c in main.columns[1:]:
        if "↓" in c or "↑" in c:
            v = main[c] if "↑" in c else -main[c]
            best[c] = np.isclose(v, v.max()) & v.notna()
    fmt = {c: "{:.1%}" if c.startswith("Lowest group share") else "{:g}" if "median rank" in c
           else "{:d}" if "claims" in c else "{:.3f}" for c in main.columns[1:]}
    t = main.copy()
    for c, f in fmt.items():
        t[c] = ["" if pd.isna(v) else f"<b>{f.format(v)}</b>" if b else f.format(v)
                for v, b in zip(main[c], best[c])]
    t.columns = pd.MultiIndex.from_tuples(groups)
    cell = "padding:4px 10px;border-bottom:1px solid rgba(128,128,128,0.35);"
    styles = [{"selector": "", "props": "border-collapse:collapse;font-size:14px;margin:6px 0"},
              {"selector": "th", "props": cell + "text-align:center;font-weight:600"},
              {"selector": "th.level0:not(:empty)",
               "props": "border-bottom:2px solid rgba(128,128,128,0.6)"},
              {"selector": "td", "props": cell + "text-align:right"},
              {"selector": "td.col0", "props": "text-align:left;white-space:nowrap"}]
    html = (t.style.set_uuid("summary").hide(axis="index").set_table_styles(styles)
            .to_html(escape=False))
    return f"<div style='overflow-x:auto'>{html}</div>"


@lru_cache(maxsize=256)
def leaderboard(config, k):
    k = _k(k)
    per, terms, rf_terms, attr_name = _lb_rows(config, k)
    agg = {c: "mean" for c in per if c not in ("model", "prompt") and not c.endswith(" p")}
    agg.update({"COCO median rank ↓": "median", "Strong claims": "sum", "Suggestive claims": "sum"})
    main = per.groupby("model", sort=False).agg(agg).reset_index()
    main["model"] = main.model.map(_short)
    cfg = LEADERBOARD[config]
    n_spread = int(per[per.model == MODELS[0]]["Excess spread ↓"].notna().sum())
    rf_text = (f"ranking fairness over {len(rf_terms)} prompts with balanced 2×2 pools "
               f"(*{attr_name}*), k = {k}" if attr_name else
               "no ranking fairness design for this attribute")
    kind = "single-person" if cfg["design"] == "single" else "couple"
    info = (f"**{config}**: utility over {len(terms)} {kind} prompts; difference in performance "
            f"with grouping *{cfg['grouping']}* (spread over the {n_spread} prompts with ≥ 2 groups "
            f"of n ≥ {MIN_N}); {rf_text}.")
    detail = per.copy()
    detail["model"] = detail.model.map(_short)
    detail = detail[["prompt"] + [c for c in per if c != "prompt"]]
    pcols = [c for c in detail if c.endswith(" p")]
    return info, _lb_html(main, k), _style(detail.reset_index(drop=True), pcols)


# ---- Difference in performance --------------------------------------------------------------------

DIP_INFO = f"""Per-image utility of each relevant image, summarized per group (computed privately; only
the group summaries are published).

* **In-dataset AUC**: fraction of the images collected for *other* prompts that score below the
  image (same-prompt images are not negatives, so groups do not compete). Chance = 0.5.
* **COCO rank**: rank of the image when inserted alone into the {DIPC['coco_images']:,} COCO val2017
  images (1 = best). Chance median = {(DIPC['coco_images'] + 2) / 2:.0f}.
* Groups are summarized by the **median**, with a distribution-free 95% CI from order statistics.
  **Δ median vs rest** = group median − median of the other labelled images of the prompt; **p**:
  two-sided permutation test of the group labels ({DIPC['n_perm']:,} permutations).
* Only groups with n ≥ {MIN_N} are published and assessed."""

DIP_P_LEGEND = P_LEGEND.format("two-sided permutation p of the group-vs-rest median difference; "
                               "COCO p Holm is corrected over the published groups of the "
                               "grouping. The *evidence* column combines them.")
SORT_KEYS = {"COCO median rank": ("rank median", True), "group size n": ("n", False)}


def _dip_table(t):
    ci = lambda lo, hi, f: [f"[{f(a)}, {f(b)}]" for a, b in zip(t[lo], t[hi])]
    out = pd.DataFrame({"group": t.group, "n": t.n, "evidence": t.evidence,
                        "direction": t.direction, "other models": t["other models"]})
    out["AUC median"] = t["AUC median"]
    out["AUC median 95% CI"] = ci("AUC CI low", "AUC CI high", lambda v: f"{v:.3f}")
    out["AUC Δ median vs rest"] = t["AUC Δ median"]
    out["AUC p"] = t["AUC p"]
    out["COCO median rank"] = t["rank median"]
    out["COCO median rank 95% CI"] = ci("rank CI low", "rank CI high", lambda v: f"{v:g}")
    out["COCO Δ median rank vs rest"] = t["rank Δ median"]
    out["COCO p"] = t["rank p"]
    out["COCO p Holm"] = t["rank p Holm"]
    out["COCO mean rank"] = t["rank mean"]
    for k in (1, 5, 10):
        out[f"COCO R@{k}"] = t[f"R@{k}"]
    return out.reset_index(drop=True)


def _dip_plot(t):
    groups = list(t.group)
    fig, axes = plt.subplots(1, 2, figsize=(12, max(2.4, 0.45 * len(groups) + 1.2)), sharey=True)
    panels = [(axes[0], "AUC", "In-dataset AUC (chance = 0.5)"),
              (axes[1], "rank", f"COCO rank (of {DIPC['coco_images'] + 1}; log scale)")]
    for ax, key, title in panels:
        for y, row in enumerate(t.to_dict("records")):
            lo, hi, med = row[f"{key} CI low"], row[f"{key} CI high"], row[f"{key} median"]
            color = label_color(row["group"])
            ax.plot([lo, hi], [y, y], color=color, lw=6, alpha=0.55, solid_capstyle="butt")
            ax.plot([med, med], [y - 0.3, y + 0.3], color="black", lw=2.2)
            p = row[f"{key} p"]
            ax.text(1.01, y, f"p={p:.3f}" + (f" · {row['evidence']}" if key == "rank" else ""),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                    fontweight="bold" if p < 0.05 else "normal")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].axvline(0.5, color="#7F7F7F", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels([f"{g} (n={n})" for g, n in zip(groups, t.n)], fontsize=9)
    axes[0].invert_yaxis()
    fig.text(0.99, 0.01, "black = group median · bar = its 95% CI (order statistics) · p: "
             "permutation test, group median vs rest of the prompt (bold < 0.05)", ha="right",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 0.9, 1))
    return fig


def group_metrics(model, prompt, grouping, sort_by):
    t = GROUPS[(GROUPS.model == model) & (GROUPS.prompt == prompt) & (GROUPS.grouping == grouping)]
    col, ascending = SORT_KEYS[sort_by]
    t = t.sort_values([col, "n"], ascending=[ascending, False], kind="stable").reset_index(drop=True)
    n_img = int(UTILITY[(UTILITY.model == model) & (UTILITY.prompt == prompt)].images.iloc[0])
    info = (f"**{prompt}** · {n_img} relevant images · grouping *{grouping}*: "
            f"{len(t)} groups with n ≥ {MIN_N}.")
    if t.empty:
        return info + " Nothing to show.", pd.DataFrame(), None
    plt.close("all")
    return info, _style(_dip_table(t), ["AUC p", "COCO p", "COCO p Holm"]), _dip_plot(t)


# ---- Evidence -------------------------------------------------------------------------------------

EVIDENCE_ORDER = ["strong", "suggestive", "protocols disagree", "uncertain"]


def evidence_all(model, groupings, show_uncertain):
    t = GROUPS[(GROUPS.model == model) & GROUPS.grouping.isin(groupings or [])]
    if t.empty:
        return "No assessed groups.", pd.DataFrame()
    counts = t.evidence.value_counts()
    summary = " · ".join(f"**{e}**: {counts.get(e, 0)}" for e in EVIDENCE_ORDER)
    if not show_uncertain:
        t = t[t.evidence != "uncertain"]
    t = t.assign(_o=t.evidence.map(EVIDENCE_ORDER.index)).sort_values(["_o", "rank p"])
    out = pd.DataFrame({
        "evidence": t.evidence, "prompt": t.prompt, "grouping": t.grouping, "group": t.group,
        "n": t.n, "direction": t.direction, "COCO median rank": t["rank median"],
        "COCO Δ median rank vs rest": t["rank Δ median"], "COCO p": t["rank p"],
        "COCO p Holm": t["rank p Holm"], "AUC Δ median vs rest": t["AUC Δ median"],
        "AUC p": t["AUC p"], "COCO R@10": t["R@10"], "other models": t["other models"]})
    return (f"{counts.sum()} assessed claims ({len(PROMPTS)} prompts × {len(groupings)} groupings, "
            f"groups with n ≥ {MIN_N}, model *{model}*): {summary}",
            _style(out.reset_index(drop=True), ["COCO p", "COCO p Holm", "AUC p"]))


# ---- Ranking fairness -----------------------------------------------------------------------------

RF_INFO = f"""Complementary analysis with ranking fairness metrics on a balanced 2×2 design:

* **single-person prompts**: perceived gender (more maleness / more femaleness) × perceived skin
  tone (FST 1 / FST 2-6);
* **couple prompts**: gender composition (different / same perceived gender) × skin tone
  composition (both people FST 1-2 / other).

Prompts are analysed when their smallest cell has ≥ {RFC['min_cell']} images.

* **Balanced pools**: each of {RFC['n_draws']:,} draws takes {RFC['per_cell']} random images from
  every cell (a pool of {4 * RFC['per_cell']}), ranks them by the model's score for the prompt, and
  measures the top-k. Values are means over draws; every model sees the same draws.
* **NDKL@k**: rank-discounted mean KL divergence of the prefix group distributions from the uniform
  target (0 = every prefix balanced). **MaxSkew@k**: max over groups of ln(share in top-k /
  target). **Normalized entropy@k**: entropy of the top-k shares / ln(#groups), 1 = balanced.
* **These metrics are not 0 for a random ordering** (e.g. 2×2 MaxSkew@12 ≈ 0.40), so each value is
  shown against its **random-order expectation** and a **permutation p** (cell labels shuffled
  within the prompt, {RFC['n_perm']:,} times). p is one-sided for the metrics (less fair than
  random) and two-sided for the shares.
* **Excess unfairness**: observed − expected for NDKL and MaxSkew, expected − observed for
  normalized entropy: 0 = as fair as random order, + = less fair. ↓ / ↑: lower / higher is fairer.
* k is a multiple of 4 so that a perfectly balanced top-k exists."""

RF_P_LEGEND = P_LEGEND.format("metrics: one-sided (less fair than random order); shares: "
                              "two-sided.")
EXCESS = "excess unfairness"


def _rf_metric_table(model, prompts, k):
    t = RF_METRIC_T[(RF_METRIC_T.model == model) & RF_METRIC_T.prompt.isin(prompts)
                    & (RF_METRIC_T.k == k)].copy()
    t[EXCESS] = t.metric.map(_sign) * t["observed − expected"]
    t["metric"] = t.metric.map(lambda m: _metric_label(m, k))
    keep = ["prompt", "attribute", "metric", "observed", "random-order expectation", EXCESS,
            "p (one-sided)", "null 95% range low", "null 95% range high", "per-draw 2.5%",
            "per-draw 97.5%"]
    return t[keep if len(prompts) > 1 else keep[1:]].reset_index(drop=True)


_COUPLE_SKIN = (_mix(FST_COLORS["1"], FST_COLORS["2"]), _mix(*(FST_COLORS[c] for c in "3456")))
RF_COLORS = {  # design -> colors of the levels of the first and the second factor
    "single": ((GENDER_COLORS["0"], GENDER_COLORS["2"]), (FST_COLORS["1"], LABEL_COLORS["FST 2-6"])),
    "couple": ((_mix(GENDER_COLORS["0"], GENDER_COLORS["2"]), "#009E73"), _COUPLE_SKIN),
}


def _rf_plot(model, prompt, k):
    design = DESIGN[prompt]
    labels = RF_DESIGNS[design]
    curves = RF_CURVES[(RF_CURVES.model == model) & (RF_CURVES.prompt == prompt)]
    attrs = list(labels["names"])
    fig, axes = plt.subplots(len(attrs), len(RF_METRICS) + 1, figsize=(15, 8.4), sharex=True)
    k_max = int(curves.k.max())
    for i, attr in enumerate(attrs):
        name = labels["names"][attr]
        for j, m in enumerate(RF_METRICS):
            c = curves[(curves.attribute == name) & (curves.series == m)].sort_values("k")
            ax = axes[i, j]
            ax.fill_between(c.k, c["null low"], c["null high"], color="#7F7F7F", alpha=0.25, lw=0)
            ax.plot(c.k, c["null mean"], color="#7F7F7F", ls="--", lw=1)
            ax.plot(c.k, c.observed, color="black", lw=1.6)
            ax.plot(c.k[3::4], c.observed[3::4], "o", color="black", ms=3)
            ax.axvline(k, color="#D55E00", lw=1, ls=":")
            if i == 0:
                ax.set_title(f"{m}@k", fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
            ax.grid(alpha=0.3)
        ax = axes[i, -1]
        groups = labels["groups"][attr]
        first, second = RF_COLORS[design]
        band = None
        for g, group in enumerate(groups):
            c = curves[(curves.attribute == name) & (curves.series == f"share: {group}")]
            c = c.sort_values("k")
            color = second[g] if attr == "b" else first[[0, 0, 1, 1][g] if attr == "ab" else g]
            ls = "--" if attr == "ab" and g % 2 else "-"
            ax.plot(c.k, c.observed, color=color, ls=ls, lw=1.8, label=group)
            band = c
        ax.fill_between(band.k, band["null low"], band["null high"], color="#7F7F7F", alpha=0.2,
                        lw=0)
        ax.axhline(1 / len(groups), color="#7F7F7F", ls=":", lw=1)
        ax.axvline(k, color="#D55E00", lw=1, ls=":")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title("share of top-k", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("k")
        ax.set_xticks(range(4, k_max + 1, 4))
    fig.text(0.99, 0.005, "black = observed (mean over balanced draws) · dashed grey = random-order "
             "expectation, band = its 95% range (permutation null) · dotted grey = balanced share · "
             f"dotted vermillion = selected k · 2×2 shares: dashed = {labels['groups']['b'][1]}",
             ha="right", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def ranking_fairness(model, prompt, k):
    k = _k(k)
    cells = RF_CELLS[RF_CELLS.prompt == prompt]
    info = (f"**{prompt}** · {DESIGN[prompt]} design · {int(cells.n.sum())} images in the 2×2 ("
            + ", ".join(f"{c}: {n}" for c, n in zip(cells.cell, cells.n))
            + f") · {RFC['n_draws']:,} balanced pools of {4 * RFC['per_cell']} · k = {k}")
    shares = RF_SHARES[(RF_SHARES.model == model) & (RF_SHARES.prompt == prompt)
                       & (RF_SHARES.k == k)]
    shares = shares[["attribute", "group", "share", "balanced share", "share − balanced",
                     "p (two-sided)"]].rename(columns={"share": f"share of top-{k}"})
    plt.close("all")
    return (info, _style(_rf_metric_table(model, [prompt], k), ["p (one-sided)"]),
            _style(shares.reset_index(drop=True), ["p (two-sided)"]), _rf_plot(model, prompt, k))


def ranking_fairness_all(model, k):
    return _style(_rf_metric_table(model, RF_PROMPTS, _k(k)), ["p (one-sided)"])


# ---- Paired comparison ----------------------------------------------------------------------------

def compare_models(model_a, model_b, k):
    k = _k(k)
    if model_a == model_b:
        return pd.DataFrame({"message": ["Pick two different models."]})
    flip = MODELS.index(model_a) > MODELS.index(model_b)
    a, b = (model_b, model_a) if flip else (model_a, model_b)
    p = PAIRED[(PAIRED["model A"] == a) & (PAIRED["model B"] == b) & (PAIRED.k == k)]
    obs = RF_METRIC_T[RF_METRIC_T.k == k].set_index(["model", "prompt", "attribute", "metric"])
    rows = []
    for r in p.to_dict("records"):
        key = (r["prompt"], r["attribute"], r["metric"])
        diff = -r["A − B"] if flip else r["A − B"]
        rows.append({"prompt": r["prompt"], "attribute": r["attribute"],
                     "metric": _metric_label(r["metric"], k),
                     f"A: {_short(model_a)}": obs.loc[(model_a, *key), "observed"],
                     f"B: {_short(model_b)}": obs.loc[(model_b, *key), "observed"],
                     "random-order expectation": obs.loc[(model_a, *key), "random-order expectation"],
                     "unfairness A − B": _sign(r["metric"]) * diff,
                     "p (paired, two-sided)": r["p"]})
    return _style(pd.DataFrame(rows), ["p (paired, two-sided)"])


# ---- UI -------------------------------------------------------------------------------------------

ABOUT = f"""### About

**InterVisions** is a benchmark of how text-to-image retrieval models treat people of different
perceived gender, age and skin tone. It contains {META['images']:,} web images collected for
{len(PROMPTS)} natural-language prompts about people in public spaces, each annotated for the
perceived attributes of the people it depicts (release {META['release']}). All labels are
perceptions of presentation by an annotator, not self-identified attributes.

**What is published here.** Only aggregate numbers: per model, prompt and group, the summaries of
the two analyses. The images, their URLs, the annotations and the embeddings are not public, and no
number in this Space refers to a single image. Groups with fewer than {MIN_N} images are neither
shown nor assessed. The evaluation code (groupings, metrics, evidence rules, ranking fairness
design) is in `evaluation/`.

**Prompts** ({len(PROMPTS)}; ranking fairness needs ≥ {RFC['min_cell']} images per 2×2 cell):

""" + "\n".join(f"* {p['prompt']} ({p['images']} images, {p['design']}"
                f"{', ranking fairness' if p['ranking_fairness'] else ''})" for p in META["prompts"]) + """

### Submit a model

Submissions are evaluated by the maintainers on the private data, through a manual queue. Open a
discussion in this Space (or an issue in its repository) with:

* the model name and a public checkpoint (an `open_clip` architecture and pretrained tag, or a
  Hugging Face model id);
* how to compute image and text embeddings (the image preprocessing and the tokenizer), if it is
  not an `open_clip` model;
* a link to the paper or model card.

Only models with public weights are accepted, in **safetensors** format or as a PyTorch state dict
that loads with `torch.load(weights_only=True)`. Pickled models and TorchScript archives are
refused, because loading them can run arbitrary code. The maintainers compute the model's
embeddings and results privately and add its aggregate numbers to this leaderboard.

### More

* The dataset in aggregate (prompts, counts, the perceived attributes and their limits):
  `docs/datasheet.md`; the annotation codes: `evaluation/codebook.md`.
* The dataset was collected and annotated with the InterVisions annotation tool:
  [github.com/InterVisions/intervisions_annotation](https://github.com/InterVisions/intervisions_annotation).
* License: code MIT; data (the published tables, datasheet and codebook) CC BY 4.0.
* Integrity: SHA-256 of the private annotation file of this release:
  `""" + META["annotations_sha256"] + """`."""

_EMBLEM = base64.b64encode((ROOT / "eu-funded.png").read_bytes()).decode()
FUNDING = (
    '<div style="display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:0.85em;'
    'margin-top:12px">'
    f'<img src="data:image/png;base64,{_EMBLEM}" alt="Co-funded by the European Union" '
    'style="height:48px;width:auto;background:#fff;padding:4px">'
    '<span style="flex:1;min-width:240px">This work is part of the InterVisions project '
    '(101214711 — CERV-2024-CHAR-LITI). Funded by the European Union. Views and opinions '
    'expressed are however those of the author(s) only and do not necessarily reflect those of '
    'the European Union or the European Education and Culture Executive Agency (EACEA). Neither '
    'the European Union nor EACEA can be held responsible for them.</span></div>')

HEADER_HELP = [
    (r"^prompt$", "Prompt (text query) the images were collected for."),
    (r"^grouping$", "Demographic grouping used to split the prompt's images."),
    (r"^group$", "Group of images (by perceived attributes) within the prompt."),
    (r"^model$", "Model (architecture / pretraining data)."),
    (r"^n$", "Number of images of the group in this prompt (only n ≥ %d are published)." % MIN_N),
    (r"^evidence$", "Evidence level of the claim: strong, suggestive, protocols disagree or "
     "uncertain (see Evidence)."),
    (r"^direction$", "Whether the group's median COCO rank is worse (ranked lower) or better "
     "(ranked higher) than that of the other labelled images of the prompt."),
    (r"^other models$", "Number of the other models where the same group goes the same way with "
     "raw COCO p < 0.05 (replication)."),
    (r"^AUC median$", "Median over the group's images of the in-dataset AUC (higher = better "
     "retrieved)."),
    (r"95% CI$", "Distribution-free 95% CI of the median from order statistics (min–max range "
     "up to n = 8)."),
    (r"^AUC Δ median vs rest$", "Median AUC of the group − median AUC of the other labelled "
     "images of the prompt (+ = group better retrieved)."),
    (r"^AUC p$", "Two-sided permutation p of the AUC median difference."),
    (r"^COCO median rank$", "Median over the group's images of the image's rank among the COCO "
     "val2017 distractors (1 = best)."),
    (r"^COCO Δ median rank vs rest$", "Median COCO rank of the group − that of the other "
     "labelled images of the prompt (+ = group ranked lower, i.e. worse)."),
    (r"^COCO p$", "Two-sided permutation p of the COCO median-rank difference."),
    (r"^COCO p Holm$", "COCO p after Holm correction over the published groups of this grouping, "
     "prompt and model."),
    (r"^COCO mean rank$", "Mean COCO rank of the group (for reference)."),
    (r"^COCO R@\d+$", "Fraction of the group's images in the top k among the COCO distractors."),
    (r"^attribute$", "Grouping of the balanced 2×2 pool: the first factor, the second, or the "
     "four cells (2×2)."),
    (r"^metric$", "Ranking fairness metric at cutoff k; ↓ = lower is fairer, ↑ = higher is "
     "fairer."),
    (r"^observed$", "Metric of the model's top-k, mean over the balanced pools."),
    (r"^random-order expectation$", "Mean of the metric under the permutation null: its value "
     "for a random ordering."),
    (r"^excess unfairness$", "Observed − expected for NDKL and MaxSkew, expected − observed for "
     "normalized entropy. 0 = as random, + = less fair."),
    (r"^p \(one-sided\)$", "Permutation p of the model being less fair than random order."),
    (r"^null 95% range (low|high)$", "2.5 / 97.5 percentiles of the metric under the "
     "permutation null."),
    (r"^per-draw (2\.5|97\.5)%$", "2.5 / 97.5 percentiles of the metric over the balanced draws "
     "(subsampling spread, not a CI)."),
    (r"^share of top-\d+$", "Share of the top-k held by the group, mean over the balanced pools."),
    (r"^balanced share$", "Share of a perfectly balanced top-k: 1 / number of groups."),
    (r"^share − balanced$", "Share of top-k − balanced share (+ = over-represented)."),
    (r"^p \(two-sided\)$", "Two-sided permutation p of the share's deviation from balance."),
    (r"^Utility$", "How well the model retrieves the prompt's images at all."),
    (r"^Difference in performance$", "Whether some groups are retrieved worse than others."),
    (r"^Ranking fairness", "Group representation in the top-k of balanced 2×2 pools, relative "
     "to random order."),
    (r"^COCO median rank ↓$", "Median COCO rank of the relevant images; summary: median over "
     "prompts. Lower = better."),
    (r"^COCO R@10 ↑$", "Fraction of the relevant images in the top 10 among the COCO "
     "distractors; summary: mean over prompts."),
    (r"^Excess spread ↓$", "log2(median COCO rank of the worst group / that of the best group) − "
     "its permutation-null expectation; summary: mean over prompts. 0 = as random labels."),
    (r"^spread p$", "One-sided permutation p of the spread being larger than for random labels."),
    (r"^Strong claims$", "Number of strong single-prompt claims of this grouping, summed over "
     "prompts."),
    (r"^Suggestive claims$", "Number of suggestive single-prompt claims of this grouping, summed "
     "over prompts."),
    (r"^(NDKL|MaxSkew)(@\d+)? excess ↓$", "Observed − random-order expectation of the metric "
     "at k (+ = less fair than random); summary: mean over the analysed prompts."),
    (r"^Entropy(@\d+)? deficit ↓$", "Random-order expectation − observed normalized entropy at "
     "k (+ = less fair than random); summary: mean over the analysed prompts."),
    (r"^(NDKL|MaxSkew|Entropy)@\d+ p$", "One-sided permutation p of the model being less fair "
     "than random order."),
    (r"^Lowest group share", "Share of the top-k held by the least represented group; summary: "
     "mean over prompts. Balanced value in brackets."),
    (r"^[AB]: ", "Observed metric of the model, mean over the balanced pools."),
    (r"^unfairness A − B$", "Metric of A − metric of B, with the sign flipped for entropy: + = "
     "A less fair than B."),
    (r"^p \(paired, two-sided\)$", "Paired permutation p of A − B (per image, the "
     "within-prompt ranks under A and B swapped at random, %d permutations)."
     % RFC["n_perm_paired"]),
]

HEADER_HELP_JS = """() => {
  const help = %s.map(([p, t]) => [new RegExp(p), t]);
  let queued = false;
  const apply = () => {
    queued = false;
    document.querySelectorAll("th").forEach((th) => {
      const text = th.innerText.replace(/\\s+/g, " ").trim();
      if (th.dataset.helpFor === text) return;
      th.dataset.helpFor = text;
      const hit = help.find(([re]) => re.test(text));
      if (hit) { th.title = hit[1]; th.style.cursor = "help"; } else th.removeAttribute("title");
    });
  };
  new MutationObserver(() => { if (!queued) { queued = true; requestAnimationFrame(apply); } })
    .observe(document.body, {childList: true, subtree: true, characterData: true});
  apply();
}""" % json.dumps(HEADER_HELP)


def build_ui():
    # Gradio 6 takes the page's JavaScript in launch(), Gradio 5 in Blocks().
    with gr.Blocks(title="InterVisions leaderboard",
                   **({} if GRADIO_6 else {"js": HEADER_HELP_JS})) as demo:
        gr.Markdown(f"## InterVisions: fairness of text→image retrieval\n{len(MODELS)} models · "
                    f"{len(PROMPTS)} prompts · {META['images']:,} annotated images (not public) · "
                    f"release {META['release']}. Aggregate results only; see *About*.")
        model = gr.Dropdown(MODELS, value=MODELS[0], label="Model (for every tab except "
                            "Leaderboard)")
        k_slider = lambda: gr.Slider(min(KS), max(KS), value=12, step=KS[1] - KS[0],
                                     label="k (ranking fairness)")

        with gr.Tab("Leaderboard"):
            gr.Markdown(LB_WARNING)
            with gr.Row():
                l_config = gr.Dropdown(list(LEADERBOARD), value=list(LEADERBOARD)[0],
                                       label="Configuration")
                l_k = k_slider()
            l_info = gr.Markdown()
            gr.Markdown("↓ / ↑: lower / higher is better; **bold** = best model in that column "
                        "(not tested).")
            l_table = gr.HTML()
            with gr.Accordion("How to read the columns", open=False):
                gr.Markdown(LB_INFO)
            with gr.Accordion("Per-prompt values behind this table", open=False):
                l_detail = gr.Dataframe(wrap=True, label="One row per prompt and model")
            for comp in (l_config, l_k):
                comp.change(leaderboard, [l_config, l_k], [l_info, l_table, l_detail])
            demo.load(leaderboard, [l_config, l_k], [l_info, l_table, l_detail])

            with gr.Accordion("Compare two models (paired test)", open=False):
                gr.Markdown(f"Paired permutation test on the same balanced pools: per image, the "
                            f"within-prompt ranks under A and B are swapped at random (H0: both "
                            f"models order the images alike), {RFC['n_perm_paired']} permutations. "
                            "Unfairness A − B > 0: A less fair than B (sign flipped for entropy).")
                with gr.Row():
                    c_a = gr.Dropdown(MODELS, value=MODELS[0], label="Model A")
                    c_b = gr.Dropdown(MODELS, value=MODELS[-1], label="Model B")
                    c_k = k_slider()
                gr.Markdown(P_LEGEND.format("two-sided paired test."))
                c_pair = gr.Dataframe(wrap=True, label="Ranking fairness metrics at k: A vs B")
            for comp in (c_a, c_b, c_k):
                comp.change(compare_models, [c_a, c_b, c_k], c_pair)
            demo.load(compare_models, [c_a, c_b, c_k], c_pair)

        with gr.Tab("Difference in performance"):
            with gr.Accordion("Analysis", open=False):
                gr.Markdown(DIP_INFO)
            with gr.Row():
                m_prompt = gr.Dropdown(PROMPTS, value=PROMPTS[0], label="Prompt")
                m_group = gr.Dropdown(GROUPINGS, value="Gender", label="Grouping")
                m_sort = gr.Radio(list(SORT_KEYS), value="COCO median rank",
                                  label="Order groups by (best first)")
            m_info = gr.Markdown()
            gr.Markdown(DIP_P_LEGEND)
            m_table = gr.Dataframe(wrap=True, label="Per-group metrics")
            m_plot = gr.Plot(label="Group medians and 95% CIs")
            m_inputs = [model, m_prompt, m_group, m_sort]
            for comp in m_inputs:
                comp.change(group_metrics, m_inputs, [m_info, m_table, m_plot])
            demo.load(group_metrics, m_inputs, [m_info, m_table, m_plot])

        with gr.Tab("Evidence"):
            gr.Markdown(EVIDENCE_RULES + f"\n\nOnly groups with n ≥ {MIN_N} are published and "
                        "assessed.")
            with gr.Row():
                e_groups = gr.Dropdown(GROUPINGS, multiselect=True, label="Groupings",
                                       value=META["default_groupings"])
                e_unc = gr.Checkbox(value=False, label="Also list uncertain claims")
            e_info = gr.Markdown()
            e_table = gr.Dataframe(wrap=True, label="Single-prompt claims, strongest first")
            e_inputs = [model, e_groups, e_unc]
            for comp in e_inputs:
                comp.change(evidence_all, e_inputs, [e_info, e_table])
            demo.load(evidence_all, e_inputs, [e_info, e_table])

        with gr.Tab("Ranking fairness"):
            with gr.Accordion("Design and metrics", open=False):
                gr.Markdown(RF_INFO)
            with gr.Row():
                f_prompt = gr.Dropdown(RF_PROMPTS, value=RF_PROMPTS[0],
                                       label=f"Prompt (smallest 2×2 cell ≥ {RFC['min_cell']})")
                f_k = k_slider()
            f_info = gr.Markdown()
            gr.Markdown(RF_P_LEGEND)
            f_table = gr.Dataframe(wrap=True, label="Metrics at k vs random-order expectation")
            f_shares = gr.Dataframe(wrap=True, label="Share of each group in the top-k")
            f_plot = gr.Plot(label="Metrics and shares as a function of k")
            f_inputs = [model, f_prompt, f_k]
            for comp in f_inputs:
                comp.change(ranking_fairness, f_inputs, [f_info, f_table, f_shares, f_plot])
            demo.load(ranking_fairness, f_inputs, [f_info, f_table, f_shares, f_plot])
            with gr.Accordion("All analysed prompts", open=False):
                f_all = gr.Dataframe(wrap=True, label="Metrics at k, all analysed prompts")
            for comp in (model, f_k):
                comp.change(ranking_fairness_all, [model, f_k], f_all)
            demo.load(ranking_fairness_all, [model, f_k], f_all)

        with gr.Tab("About and submissions"):
            gr.Markdown(ABOUT)
        gr.HTML(FUNDING)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None, help="default: 7860 or $GRADIO_SERVER_PORT")
    ap.add_argument("--host", default=None, help="default: 127.0.0.1 or $GRADIO_SERVER_NAME")
    args = ap.parse_args()
    build_ui().queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=args.port,
        **({"js": HEADER_HELP_JS} if GRADIO_6 else {}))
