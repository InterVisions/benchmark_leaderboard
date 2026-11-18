// InterVisions leaderboard: static version of app.py (same views, same numbers).
// Reads data/meta.json, data/*.csv and static/config.json; no other site is contacted.
"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const fmtInt = (v) => Number(v).toLocaleString("en-US");

// Number formats exactly as Python's (pandas Styler, str.format): the binary value correctly
// rounded with ties to even, and "-" kept for negatives that round to zero ("-0.000").
function pyFixed(x, d) {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x > 0 ? "inf" : "-inf";
  const neg = x < 0 || Object.is(x, -0);
  const [ip, fp] = Math.abs(x).toFixed(Math.min(100, d + 30)).split(".");
  const rest = fp.slice(d);
  let digits = BigInt(ip + fp.slice(0, d));
  if (rest[0] > "5" || (rest[0] === "5" && (/[1-9]/.test(rest.slice(1)) || digits % 2n === 1n))) digits += 1n;
  let out = digits.toString();
  if (d > 0) { out = out.padStart(d + 1, "0"); out = out.slice(0, -d) + "." + out.slice(-d); }
  return (neg ? "-" : "") + out;
}
const pyPct = (x, d = 1) => pyFixed(100 * x, d) + "%";  // Python's "{:.1%}" multiplies the float by 100
function pyG(x) {  // Python's "{:g}" (6 significant digits) for the values used here
  if (!Number.isFinite(x)) return pyFixed(x, 0);
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const e = Math.floor(Math.log10(Math.abs(x)));
  if (e < -4 || e >= 6) return String(Number(x.toPrecision(6)));
  const s = pyFixed(x, Math.max(0, 5 - e));
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

// ---- Data ---------------------------------------------------------------------------------------

function parseCSV(text) {
  // RFC 4180 (pandas to_csv): quoted fields may hold commas, quotes ("") and newlines.
  const rows = [];
  let row = [], field = "", i = 0, quoted = false;
  while (i < text.length) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
    i++;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift();
  const numeric = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
  return rows.filter((r) => r.length === header.length).map((r) => {
    const o = {};
    header.forEach((h, j) => {
      const v = r[j];
      o[h] = v === "" ? null : numeric.test(v) ? Number(v) : v === "inf" ? Infinity : v === "-inf" ? -Infinity : v;
    });
    return o;
  });
}

const cache = {};
function load(name) {
  if (!cache[name]) {
    const url = name.endsWith(".json") ? name : `data/${name}.csv`;
    cache[name] = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return name.endsWith(".json") ? r.json() : r.text().then(parseCSV);
    });
  }
  return cache[name];
}

let META, CONF, MODELS, PROMPTS, DESIGN, RF_PROMPTS, KS, MIN_N, RFC, DIPC;

// ---- Texts (as in app.py) ------------------------------------------------------------------------

const P_LEGEND = (extra) => "**p shading**: <span style='background:#D55E00;color:white;padding:0 4px'>p &lt; 0.01</span> " +
  "<span style='background:#F6C9A6;color:black;padding:0 4px'>p &lt; 0.05</span> · " + extra +
  " Not corrected for multiple comparisons unless stated, so an isolated p &lt; 0.05 is a hint, not a finding.";

function texts() {
  const n = (v) => fmtInt(v);
  return {
    LB_WARNING: `**Summary only.** Each number is an average over a handful of prompts and hides
which group is favoured and where; a single number is not a verdict on a model, and the columns are
not combined into a score. Fairness columns are **excess over chance** (random labels or random
order): 0 = as fair as chance. Read them next to utility: a model can look fair by being bad for
everyone. Differences between two models are tested only in *Compare two models*. Groups with
fewer than ${MIN_N} images are not published and not assessed.`,
    LB_INFO: `* **COCO median rank** (median over prompts of the median rank of the relevant images among
  ${n(DIPC.coco_images)} COCO val2017 distractors) and **COCO R@10** (mean over prompts): utility.
* **Excess spread** (difference in performance): per prompt, log2 of the median COCO rank of the
  worst group (n ≥ ${MIN_N}) over that of the best group, minus its permutation-null expectation;
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
  prompts, both FST 1-2 / other for couples).`,
    CMP_INFO: `Paired permutation test on the same balanced pools: per image, the within-prompt ranks under A
and B are swapped at random (H0: both models order the images alike), ${RFC.n_perm_paired}
permutations. Unfairness A − B > 0: A less fair than B (sign flipped for entropy).`,
    P_LEGEND_PAIRED: P_LEGEND("two-sided paired test."),
    DIP_INFO: `Per-image utility of each relevant image, summarized per group (computed privately; only
the group summaries are published).

* **In-dataset AUC**: fraction of the images collected for *other* prompts that score below the
  image (same-prompt images are not negatives, so groups do not compete). Chance = 0.5.
* **COCO rank**: rank of the image when inserted alone into the ${n(DIPC.coco_images)} COCO val2017
  images (1 = best). Chance median = ${Math.round((DIPC.coco_images + 2) / 2)}.
* Groups are summarized by the **median**, with a distribution-free 95% CI from order statistics.
  **Δ median vs rest** = group median − median of the other labelled images of the prompt; **p**:
  two-sided permutation test of the group labels (${n(DIPC.n_perm)} permutations).
* Only groups with n ≥ ${MIN_N} are published and assessed.`,
    DIP_P_LEGEND: P_LEGEND("two-sided permutation p of the group-vs-rest median difference; COCO p Holm " +
      "is corrected over the published groups of the grouping. The *evidence* column combines them."),
    EVIDENCE: CONF.evidence_rules + `\n\nOnly groups with n ≥ ${MIN_N} are published and assessed.`,
    RF_INFO: `Complementary analysis with ranking fairness metrics on a balanced 2×2 design:

* **single-person prompts**: perceived gender (more maleness / more femaleness) × perceived skin
  tone (FST 1 / FST 2-6);
* **couple prompts**: gender composition (different / same perceived gender) × skin tone
  composition (both people FST 1-2 / other).

Prompts are analysed when their smallest cell has ≥ ${RFC.min_cell} images.

* **Balanced pools**: each of ${n(RFC.n_draws)} draws takes ${RFC.per_cell} random images from
  every cell (a pool of ${4 * RFC.per_cell}), ranks them by the model's score for the prompt, and
  measures the top-k. Values are means over draws; every model sees the same draws.
* **NDKL@k**: rank-discounted mean KL divergence of the prefix group distributions from the uniform
  target (0 = every prefix balanced). **MaxSkew@k**: max over groups of ln(share in top-k /
  target). **Normalized entropy@k**: entropy of the top-k shares / ln(#groups), 1 = balanced.
* **These metrics are not 0 for a random ordering** (e.g. 2×2 MaxSkew@12 ≈ 0.40), so each value is
  shown against its **random-order expectation** and a **permutation p** (cell labels shuffled
  within the prompt, ${n(RFC.n_perm)} times). p is one-sided for the metrics (less fair than
  random) and two-sided for the shares.
* **Excess unfairness**: observed − expected for NDKL and MaxSkew, expected − observed for
  normalized entropy: 0 = as fair as random order, + = less fair. ↓ / ↑: lower / higher is fairer.
* k is a multiple of 4 so that a perfectly balanced top-k exists.`,
    RF_P_LEGEND: P_LEGEND("metrics: one-sided (less fair than random order); shares: two-sided."),
    ABOUT: `### About

**InterVisions** is a benchmark of how text-to-image retrieval models treat people of different
perceived gender, age and skin tone. It contains ${n(META.images)} web images collected for
${PROMPTS.length} natural-language prompts about people in public spaces, each annotated for the
perceived attributes of the people it depicts (release ${META.release}). All labels are
perceptions of presentation by an annotator, not self-identified attributes.

**What is published here.** Only aggregate numbers: per model, prompt and group, the summaries of
the two analyses. The images, their URLs, the annotations and the embeddings are not public, and no
number on this page refers to a single image. Groups with fewer than ${MIN_N} images are neither
shown nor assessed. The evaluation code (groupings, metrics, evidence rules, ranking fairness
design) is in \`evaluation/\` of the repository.

**Prompts** (${PROMPTS.length}; ranking fairness needs ≥ ${RFC.min_cell} images per 2×2 cell):

` + META.prompts.map((p) => `* ${p.prompt} (${p.images} images, ${p.design}${p.ranking_fairness ? ", ranking fairness" : ""})`).join("\n") + `

### Submit a model

Submissions are evaluated by the maintainers on the private data, through a manual queue. Open a
discussion in the Hugging Face Space (or an issue in the GitHub repository) with:

* the model name and a public checkpoint (an \`open_clip\` architecture and pretrained tag, or a
  Hugging Face model id);
* how to compute image and text embeddings (the image preprocessing and the tokenizer), if it is
  not an \`open_clip\` model;
* a link to the paper or model card.

Only models with public weights are accepted, in **safetensors** format or as a PyTorch state dict
that loads with \`torch.load(weights_only=True)\`. Pickled models and TorchScript archives are
refused, because loading them can run arbitrary code. The maintainers compute the model's
embeddings and results privately and add its aggregate numbers to this leaderboard.

### More

* Code and data: [github.com/InterVisions/benchmark_leaderboard](https://github.com/InterVisions/benchmark_leaderboard).
  The dataset in aggregate (prompts, counts, the perceived attributes and their limits):
  \`docs/datasheet.md\`; the annotation codes: \`evaluation/codebook.md\`.
* The dataset was collected and annotated with the InterVisions annotation tool:
  [github.com/InterVisions/intervisions_annotation](https://github.com/InterVisions/intervisions_annotation).
* License: code MIT; data (the published tables, datasheet and codebook) CC BY 4.0.
* Integrity: SHA-256 of the private annotation file of this release:
  \`${META.annotations_sha256}\`.`,
  };
}

// Minimal Markdown for the texts above (paragraphs, lists, ###, **, *, `code`, links, inline HTML).
function md(src) {
  const inline = (s) => s
    .replace(/`([^`]+)`/g, (_, c) => `<code>${esc(c)}</code>`)
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?=[^*\w]|$)/g, "$1<i>$2</i>");
  const out = [];
  for (const block of src.trim().split(/\n\s*\n/)) {
    const lines = block.split("\n");
    if (/^###\s/.test(lines[0])) {
      out.push(`<h3>${inline(lines[0].replace(/^###\s+/, ""))}</h3>`);
      if (lines.length > 1) out.push(`<p>${inline(lines.slice(1).join(" "))}</p>`);
    } else if (/^\s*[*-]\s/.test(lines[0])) {
      const items = [];
      for (const l of lines) {
        if (/^\s*[*-]\s/.test(l)) items.push(l.replace(/^\s*[*-]\s+/, ""));
        else items[items.length - 1] += " " + l.trim();
      }
      out.push("<ul>" + items.map((t) => `<li>${inline(t)}</li>`).join("") + "</ul>");
    } else out.push(`<p>${inline(lines.join(" "))}</p>`);
  }
  return out.join("\n");
}

// ---- Tables ---------------------------------------------------------------------------------------

const HEADER_HELP = [
  [/^prompt$/, "Prompt (text query) the images were collected for."],
  [/^grouping$/, "Demographic grouping used to split the prompt's images."],
  [/^group$/, "Group of images (by perceived attributes) within the prompt."],
  [/^model$/, "Model (architecture / pretraining data)."],
  [/^n$/, () => `Number of images of the group in this prompt (only n ≥ ${MIN_N} are published).`],
  [/^evidence$/, "Evidence level of the claim: strong, suggestive, protocols disagree or uncertain (see Evidence)."],
  [/^direction$/, "Whether the group's median COCO rank is worse (ranked lower) or better (ranked higher) than that of the other labelled images of the prompt."],
  [/^other models$/, "Number of the other models where the same group goes the same way with raw COCO p < 0.05 (replication)."],
  [/^AUC median$/, "Median over the group's images of the in-dataset AUC (higher = better retrieved)."],
  [/95% CI$/, "Distribution-free 95% CI of the median from order statistics (min–max range up to n = 8)."],
  [/^AUC Δ median vs rest$/, "Median AUC of the group − median AUC of the other labelled images of the prompt (+ = group better retrieved)."],
  [/^AUC p$/, "Two-sided permutation p of the AUC median difference."],
  [/^COCO median rank$/, "Median over the group's images of the image's rank among the COCO val2017 distractors (1 = best)."],
  [/^COCO Δ median rank vs rest$/, "Median COCO rank of the group − that of the other labelled images of the prompt (+ = group ranked lower, i.e. worse)."],
  [/^COCO p$/, "Two-sided permutation p of the COCO median-rank difference."],
  [/^COCO p Holm$/, "COCO p after Holm correction over the published groups of this grouping, prompt and model."],
  [/^COCO mean rank$/, "Mean COCO rank of the group (for reference)."],
  [/^COCO R@\d+$/, "Fraction of the group's images in the top k among the COCO distractors."],
  [/^attribute$/, "Grouping of the balanced 2×2 pool: the first factor, the second, or the four cells (2×2)."],
  [/^metric$/, "Ranking fairness metric at cutoff k; ↓ = lower is fairer, ↑ = higher is fairer."],
  [/^observed$/, "Metric of the model's top-k, mean over the balanced pools."],
  [/^random-order expectation$/, "Mean of the metric under the permutation null: its value for a random ordering."],
  [/^excess unfairness$/, "Observed − expected for NDKL and MaxSkew, expected − observed for normalized entropy. 0 = as random, + = less fair."],
  [/^p \(one-sided\)$/, "Permutation p of the model being less fair than random order."],
  [/^null 95% range (low|high)$/, "2.5 / 97.5 percentiles of the metric under the permutation null."],
  [/^per-draw (2\.5|97\.5)%$/, "2.5 / 97.5 percentiles of the metric over the balanced draws (subsampling spread, not a CI)."],
  [/^share of top-\d+$/, "Share of the top-k held by the group, mean over the balanced pools."],
  [/^balanced share$/, "Share of a perfectly balanced top-k: 1 / number of groups."],
  [/^share − balanced$/, "Share of top-k − balanced share (+ = over-represented)."],
  [/^p \(two-sided\)$/, "Two-sided permutation p of the share's deviation from balance."],
  [/^Utility$/, "How well the model retrieves the prompt's images at all."],
  [/^Difference in performance$/, "Whether some groups are retrieved worse than others."],
  [/^Ranking fairness/, "Group representation in the top-k of balanced 2×2 pools, relative to random order."],
  [/^COCO median rank ↓$/, "Median COCO rank of the relevant images; summary: median over prompts. Lower = better."],
  [/^COCO R@10 ↑$/, "Fraction of the relevant images in the top 10 among the COCO distractors; summary: mean over prompts."],
  [/^Excess spread ↓$/, "log2(median COCO rank of the worst group / that of the best group) − its permutation-null expectation; summary: mean over prompts. 0 = as random labels."],
  [/^spread p$/, "One-sided permutation p of the spread being larger than for random labels."],
  [/^Strong claims$/, "Number of strong single-prompt claims of this grouping, summed over prompts."],
  [/^Suggestive claims$/, "Number of suggestive single-prompt claims of this grouping, summed over prompts."],
  [/^(NDKL|MaxSkew)(@\d+)? excess ↓$/, "Observed − random-order expectation of the metric at k (+ = less fair than random); summary: mean over the analysed prompts."],
  [/^Entropy(@\d+)? deficit ↓$/, "Random-order expectation − observed normalized entropy at k (+ = less fair than random); summary: mean over the analysed prompts."],
  [/^(NDKL|MaxSkew|Entropy)@\d+ p$/, "One-sided permutation p of the model being less fair than random order."],
  [/^Lowest group share/, "Share of the top-k held by the least represented group; summary: mean over prompts. Balanced value in brackets."],
  [/^[AB]: /, "Observed metric of the model, mean over the balanced pools."],
  [/^unfairness A − B$/, "Metric of A − metric of B, with the sign flipped for entropy: + = A less fair than B."],
  [/^p \(paired, two-sided\)$/, () => `Paired permutation p of A − B (per image, the within-prompt ranks under A and B swapped at random, ${RFC.n_perm_paired} permutations).`],
];

function th(name, attrs = "") {
  const hit = HEADER_HELP.find(([re]) => re.test(name));
  const help = hit ? (typeof hit[1] === "function" ? hit[1]() : hit[1]) : "";
  return `<th${attrs}${help ? ` title="${esc(help)}"` : ""}>${esc(name)}</th>`;
}

const INT_COLS = new Set(["n", "k", "images", "assessed groups", "Strong claims", "Suggestive claims"]);

function fmtCell(col, v) {
  // As app.py's _style: 3 decimals by default, :g for median ranks, .1f for the mean rank, % for shares.
  if (v === null || v === undefined || (typeof v === "number" && Number.isNaN(v))) return "";
  if (typeof v !== "number") return esc(v);
  if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf";
  if (/^(share|balanced share|Lowest group share)/.test(col)) return pyPct(v);
  if (col === "COCO median rank" || col === "COCO Δ median rank vs rest") return pyG(v);
  if (col === "COCO mean rank") return pyFixed(v, 1);
  if (INT_COLS.has(col)) return String(v);
  return pyFixed(v, 3);
}

function pShade(v) {
  if (!isNum(v) || v >= 0.05) return "";
  return v < 0.01 ? "background:#D55E00;color:white" : "background:#F6C9A6;color:black";
}

function groupShade(label) {
  const c = CONF.label_colors[label] || "";
  if (!c) return "";
  const dark = [1, 3, 5].reduce((s, i) => s + parseInt(c.slice(i, i + 2), 16), 0) < 3 * 140;
  return `background:${c};color:${dark ? "white" : "black"}`;
}

function table(rows, cols, {pcols = [], short = false, empty = "Nothing to show."} = {}) {
  if (!rows.length) return `<p class="note">${esc(empty)}</p>`;
  const pset = new Set(pcols);
  const head = "<tr>" + cols.map((c) => th(c)).join("") + "</tr>";
  const body = rows.map((r) => "<tr>" + cols.map((c) => {
    const v = r[c];
    const style = c === "group" ? groupShade(v) : pset.has(c) ? pShade(v) : "";
    const cls = typeof v === "number" ? "num" : "txt";
    return `<td class="${cls}"${style ? ` style="${style}"` : ""}>${fmtCell(c, v)}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="table-wrap${short ? " short" : ""}"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}

// ---- Helpers --------------------------------------------------------------------------------------

const short = (m) => m.replace(" / ", " ");
const sign = (metric) => (META.rf_metrics[metric] ? 1 : -1);  // + = less fair
const metricLabel = (metric, k) => `${metric}${k ? "@" + k : ""} ${META.rf_metrics[metric] ? "↓" : "↑"}`;
const mean = (xs) => { const v = xs.filter(isNum); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
const median = (xs) => {
  const v = xs.filter(isNum).sort((a, b) => a - b);
  if (!v.length) return null;
  const h = Math.floor(v.length / 2);
  return v.length % 2 ? v[h] : (v[h - 1] + v[h]) / 2;
};
const sum = (xs) => xs.filter(isNum).reduce((a, b) => a + b, 0);
const isClose = (a, b) => Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b);
const stableSort = (arr, cmp) => arr.map((v, i) => [v, i]).sort((a, b) => cmp(a[0], b[0]) || a[1] - b[1]).map((x) => x[0]);

function options(select, values, value) {
  select.innerHTML = values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
  if (value !== undefined) select.value = value;
}

function kSlider(id) {
  const el = $(`#${id}`);
  el.min = Math.min(...KS); el.max = Math.max(...KS); el.step = KS[1] - KS[0]; el.value = 12;
  const show = () => { $(`#${id}-val`).textContent = el.value; };
  el.addEventListener("input", show); show();
  return el;
}

// ---- Leaderboard ----------------------------------------------------------------------------------

async function leaderboard() {
  const config = $("#lb-config").value, k = Number($("#lb-k").value);
  const [UTILITY, SPREAD, GROUPS, RFM, RFS] = await Promise.all(
    ["utility", "spread", "groups", "rf_metrics", "rf_shares"].map(load));
  const cfg = META.leaderboard[config];
  const {design, attribute: attr, grouping} = cfg;
  const terms = PROMPTS.filter((t) => DESIGN[t] === design);
  const rfTerms = attr ? RF_PROMPTS.filter((t) => DESIGN[t] === design) : [];
  const attrName = attr ? META.rf_designs[design].names[attr] : null;
  const per = [], cols = [];
  const add = (row, c, v) => { row[c] = v; if (!cols.includes(c)) cols.push(c); };
  for (const m of MODELS) {
    for (const t of terms) {
      const u = UTILITY.find((r) => r.model === m && r.prompt === t);
      const s = SPREAD.find((r) => r.model === m && r.prompt === t && r.grouping === grouping);
      const ev = GROUPS.filter((r) => r.model === m && r.prompt === t && r.grouping === grouping);
      const row = {};
      add(row, "model", m); add(row, "prompt", t);
      add(row, "COCO median rank ↓", u["COCO median rank"]);
      add(row, "COCO R@10 ↑", u["COCO R@10"]);
      add(row, "Excess spread ↓", s ? s.spread - s["null mean"] : null);
      add(row, "spread p", s ? s.p : null);
      add(row, "Strong claims", ev.filter((r) => r.evidence === "strong").length);
      add(row, "Suggestive claims", ev.filter((r) => r.evidence === "suggestive").length);
      if (rfTerms.includes(t)) {
        for (const r of RFM.filter((r) => r.model === m && r.prompt === t && r.k === k && r.attribute === attrName)) {
          const col = r.metric === "Normalized entropy" ? `Entropy@${k} deficit ↓` : `${r.metric}@${k} excess ↓`;
          add(row, col, sign(r.metric) * r["observed − expected"]);
          add(row, `${col.split(" ")[0]} p`, r["p (one-sided)"]);
        }
        const sh = RFS.filter((r) => r.model === m && r.prompt === t && r.k === k && r.attribute === attrName);
        add(row, `Lowest group share@${k} ↑ (balanced ${Math.round(100 / sh.length)}%)`, Math.min(...sh.map((r) => r.share)));
      }
      per.push(row);
    }
  }
  // Summary per model: median of the COCO median rank, sums of claims, means of the rest.
  const aggCols = cols.filter((c) => c !== "model" && c !== "prompt" && !c.endsWith(" p"));
  const main = MODELS.map((m) => {
    const rs = per.filter((r) => r.model === m), o = {model: short(m)};
    for (const c of aggCols) {
      const vals = rs.map((r) => (r[c] === undefined ? null : r[c]));
      o[c] = c === "COCO median rank ↓" ? median(vals) : c.endsWith("claims") ? sum(vals) : mean(vals);
    }
    return o;
  });
  const nSpread = per.filter((r) => r.model === MODELS[0] && isNum(r["Excess spread ↓"])).length;
  const rfText = attrName ? `ranking fairness over ${rfTerms.length} prompts with balanced 2×2 pools (*${attrName}*), k = ${k}`
    : "no ranking fairness design for this attribute";
  const kind = design === "single" ? "single-person" : "couple";
  $("#lb-info").innerHTML = md(`**${config}**: utility over ${terms.length} ${kind} prompts; difference in performance ` +
    `with grouping *${grouping}* (spread over the ${nSpread} prompts with ≥ 2 groups of n ≥ ${MIN_N}); ${rfText}.`);
  $("#lb-table").innerHTML = summaryTable(main, ["model", ...aggCols], k);
  const detail = per.map((r) => ({...r, model: short(r.model)}));
  const dcols = ["prompt", ...cols.filter((c) => c !== "prompt")];
  $("#lb-detail").innerHTML = table(detail, dcols, {pcols: dcols.filter((c) => c.endsWith(" p"))});
}

function summaryTable(main, cols, k) {
  const block = (c) => (c.startsWith("COCO") ? "Utility" : c.includes(`@${k}`) ? `Ranking fairness (top-${k})` : "Difference in performance");
  const blocks = [];
  for (const c of cols.slice(1)) {
    const b = block(c);
    if (blocks.length && blocks[blocks.length - 1][0] === b) blocks[blocks.length - 1][1]++;
    else blocks.push([b, 1]);
  }
  const best = {};
  for (const c of cols.slice(1)) {
    if (!c.includes("↓") && !c.includes("↑")) continue;
    const v = main.map((r) => (isNum(r[c]) ? (c.includes("↑") ? r[c] : -r[c]) : null));
    const max = Math.max(...v.filter(isNum));
    best[c] = v.map((x) => isNum(x) && isClose(x, max));
  }
  const fmt = (c, v) => {
    if (!isNum(v)) return "";
    if (c.startsWith("Lowest group share")) return pyPct(v);
    if (c.includes("median rank")) return pyG(v);
    if (c.includes("claims")) return String(v);
    return pyFixed(v, 3);
  };
  const head1 = "<tr><th></th>" + blocks.map(([b, n]) => th(b, ` colspan="${n}"`)).join("") + "</tr>";
  const head2 = "<tr>" + cols.map((c) => th(c.replace(`@${k}`, ""))).join("") + "</tr>";
  const body = main.map((r, i) => "<tr>" + cols.map((c, j) => {
    if (j === 0) return `<td class="txt" style="white-space:nowrap">${esc(r[c])}</td>`;
    const s = fmt(c, r[c]);
    return `<td class="num">${best[c] && best[c][i] ? `<b>${s}</b>` : s}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="table-wrap short"><table class="lb"><thead>${head1}${head2}</thead><tbody>${body}</tbody></table></div>`;
}

async function compareModels() {
  const ma = $("#cmp-a").value, mb = $("#cmp-b").value, k = Number($("#cmp-k").value);
  if (ma === mb) { $("#cmp-table").innerHTML = '<p class="note">Pick two different models.</p>'; return; }
  const [PAIRED, RFM] = await Promise.all([load("paired"), load("rf_metrics")]);
  const flip = MODELS.indexOf(ma) > MODELS.indexOf(mb);
  const [a, b] = flip ? [mb, ma] : [ma, mb];
  const obs = new Map(RFM.filter((r) => r.k === k).map((r) => [[r.model, r.prompt, r.attribute, r.metric].join("\u0001"), r]));
  const key = (m, r) => [m, r.prompt, r.attribute, r.metric].join("\u0001");
  const colA = `A: ${short(ma)}`, colB = `B: ${short(mb)}`;
  const rows = PAIRED.filter((r) => r["model A"] === a && r["model B"] === b && r.k === k).map((r) => {
    const diff = flip ? -r["A − B"] : r["A − B"];
    return {prompt: r.prompt, attribute: r.attribute, metric: metricLabel(r.metric, k),
            [colA]: obs.get(key(ma, r)).observed, [colB]: obs.get(key(mb, r)).observed,
            "random-order expectation": obs.get(key(ma, r))["random-order expectation"],
            "unfairness A − B": sign(r.metric) * diff, "p (paired, two-sided)": r.p};
  });
  $("#cmp-table").innerHTML = table(rows, ["prompt", "attribute", "metric", colA, colB, "random-order expectation",
    "unfairness A − B", "p (paired, two-sided)"], {pcols: ["p (paired, two-sided)"]});
}

// ---- Difference in performance --------------------------------------------------------------------

const SORT_KEYS = {"COCO median rank": ["rank median", true], "group size n": ["n", false]};

async function groupMetrics() {
  const model = $("#model").value, prompt = $("#dip-prompt").value, grouping = $("#dip-grouping").value;
  const [col, asc] = SORT_KEYS[$("#dip-sort").value];
  const [GROUPS, UTILITY] = await Promise.all([load("groups"), load("utility")]);
  let t = GROUPS.filter((r) => r.model === model && r.prompt === prompt && r.grouping === grouping);
  t = stableSort(t, (x, y) => (asc ? x[col] - y[col] : y[col] - x[col]) || y.n - x.n);
  const nImg = UTILITY.find((r) => r.model === model && r.prompt === prompt).images;
  let info = `**${prompt}** · ${nImg} relevant images · grouping *${grouping}*: ${t.length} groups with n ≥ ${MIN_N}.`;
  if (!t.length) info += " Nothing to show.";
  $("#dip-info").innerHTML = md(info);
  const ci = (lo, hi, f) => (r) => `[${f(r[lo])}, ${f(r[hi])}]`;
  const rows = t.map((r) => ({
    group: r.group, n: r.n, evidence: r.evidence, direction: r.direction, "other models": r["other models"],
    "AUC median": r["AUC median"], "AUC median 95% CI": ci("AUC CI low", "AUC CI high", (v) => pyFixed(v, 3))(r),
    "AUC Δ median vs rest": r["AUC Δ median"], "AUC p": r["AUC p"], "COCO median rank": r["rank median"],
    "COCO median rank 95% CI": ci("rank CI low", "rank CI high", pyG)(r), "COCO Δ median rank vs rest": r["rank Δ median"],
    "COCO p": r["rank p"], "COCO p Holm": r["rank p Holm"], "COCO mean rank": r["rank mean"],
    "COCO R@1": r["R@1"], "COCO R@5": r["R@5"], "COCO R@10": r["R@10"]}));
  $("#dip-table").innerHTML = t.length ? table(rows, Object.keys(rows[0]), {pcols: ["AUC p", "COCO p", "COCO p Holm"], short: true}) : "";
  $("#dip-plot").innerHTML = t.length ? dipPlot(t) : "";
}

// ---- Evidence -------------------------------------------------------------------------------------

const EVIDENCE_ORDER = ["strong", "suggestive", "protocols disagree", "uncertain"];

async function evidenceAll() {
  const model = $("#model").value, showUnc = $("#ev-uncertain").checked;
  const groupings = [...document.querySelectorAll("#ev-groupings input:checked")].map((i) => i.value);
  const GROUPS = await load("groups");
  let t = GROUPS.filter((r) => r.model === model && groupings.includes(r.grouping));
  if (!t.length) { $("#ev-info").innerHTML = md("No assessed groups."); $("#ev-table").innerHTML = ""; return; }
  const counts = {};
  t.forEach((r) => { counts[r.evidence] = (counts[r.evidence] || 0) + 1; });
  const summary = EVIDENCE_ORDER.map((e) => `**${e}**: ${counts[e] || 0}`).join(" · ");
  if (!showUnc) t = t.filter((r) => r.evidence !== "uncertain");
  t = stableSort(t, (x, y) => EVIDENCE_ORDER.indexOf(x.evidence) - EVIDENCE_ORDER.indexOf(y.evidence) || x["rank p"] - y["rank p"]);
  const rows = t.map((r) => ({evidence: r.evidence, prompt: r.prompt, grouping: r.grouping, group: r.group, n: r.n,
    direction: r.direction, "COCO median rank": r["rank median"], "COCO Δ median rank vs rest": r["rank Δ median"],
    "COCO p": r["rank p"], "COCO p Holm": r["rank p Holm"], "AUC Δ median vs rest": r["AUC Δ median"],
    "AUC p": r["AUC p"], "COCO R@10": r["R@10"], "other models": r["other models"]}));
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  $("#ev-info").innerHTML = md(`${total} assessed claims (${PROMPTS.length} prompts × ${groupings.length} groupings, ` +
    `groups with n ≥ ${MIN_N}, model *${model}*): ${summary}`);
  $("#ev-table").innerHTML = table(rows, ["evidence", "prompt", "grouping", "group", "n", "direction", "COCO median rank",
    "COCO Δ median rank vs rest", "COCO p", "COCO p Holm", "AUC Δ median vs rest", "AUC p", "COCO R@10", "other models"],
    {pcols: ["COCO p", "COCO p Holm", "AUC p"], empty: "No claims to list (uncertain claims are hidden)."});
}

// ---- Ranking fairness -----------------------------------------------------------------------------

const EXCESS = "excess unfairness";
const RF_KEEP = ["prompt", "attribute", "metric", "observed", "random-order expectation", EXCESS, "p (one-sided)",
  "null 95% range low", "null 95% range high", "per-draw 2.5%", "per-draw 97.5%"];

function rfMetricRows(RFM, model, prompts, k) {
  return RFM.filter((r) => r.model === model && prompts.includes(r.prompt) && r.k === k)
    .map((r) => ({...r, [EXCESS]: sign(r.metric) * r["observed − expected"], metric: metricLabel(r.metric, k)}));
}

async function rankingFairness() {
  const model = $("#model").value, prompt = $("#rf-prompt").value, k = Number($("#rf-k").value);
  const [RFM, RFS, CELLS] = await Promise.all([load("rf_metrics"), load("rf_shares"), load("rf_cells")]);
  const cells = CELLS.filter((r) => r.prompt === prompt);
  $("#rf-info").innerHTML = md(`**${prompt}** · ${DESIGN[prompt]} design · ${sum(cells.map((c) => c.n))} images in the 2×2 (` +
    cells.map((c) => `${c.cell}: ${c.n}`).join(", ") +
    `) · ${fmtInt(RFC.n_draws)} balanced pools of ${4 * RFC.per_cell} · k = ${k}`);
  $("#rf-table").innerHTML = table(rfMetricRows(RFM, model, [prompt], k), RF_KEEP.slice(1), {pcols: ["p (one-sided)"], short: true});
  const shareCol = `share of top-${k}`;
  const shares = RFS.filter((r) => r.model === model && r.prompt === prompt && r.k === k)
    .map((r) => ({...r, [shareCol]: r.share}));
  $("#rf-shares").innerHTML = table(shares, ["attribute", "group", shareCol, "balanced share", "share − balanced", "p (two-sided)"],
    {pcols: ["p (two-sided)"], short: true});
  $("#rf-plot").innerHTML = '<p class="note">Loading the curves…</p>';
  const CURVES = await load("rf_curves");
  if ($("#rf-prompt").value === prompt && $("#model").value === model) $("#rf-plot").innerHTML = rfPlot(CURVES, model, prompt, k);
}

async function rankingFairnessAll() {
  const RFM = await load("rf_metrics");
  const model = $("#model").value, k = Number($("#rf-k").value);
  $("#rf-all").innerHTML = table(rfMetricRows(RFM, model, RF_PROMPTS, k), RF_KEEP, {pcols: ["p (one-sided)"]});
}

// ---- SVG plots ------------------------------------------------------------------------------------

const svgText = (x, y, s, attrs = "") => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" ${attrs}>${esc(s)}</text>`;
const line = (x1, y1, x2, y2, attrs) => `<line x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" ${attrs}/>`;

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo && v <= hi) out.push(v); }
  return out;
}

function dipPlot(t) {
  const W = 1200, labelW = 230, pW = 130, gap = 30, top = 28, rowH = 30, bottom = 54;
  const panelW = (W - labelW - 2 * pW - gap) / 2, H = top + rowH * t.length + bottom;
  const panels = [
    {key: "AUC", title: "In-dataset AUC (chance = 0.5)", log: false},
    {key: "rank", title: `COCO rank (of ${fmtInt(DIPC.coco_images + 1)}; log scale)`, log: true},
  ];
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="Group medians and 95% CIs">`;
  t.forEach((r, i) => {
    s += svgText(labelW - 8, top + rowH * (i + 0.5) + 4, `${r.group} (n=${r.n})`, 'text-anchor="end" font-size="12"');
  });
  panels.forEach((p, j) => {
    const x0 = labelW + j * (panelW + pW + gap);
    // As app.py (matplotlib): the axis spans the data; the chance line is drawn only if inside it.
    const lo = Math.min(...t.map((r) => r[`${p.key} CI low`])), hi = Math.max(...t.map((r) => r[`${p.key} CI high`]));
    let a, b, ticks, X;
    if (p.log) {
      a = Math.max(lo, 0.8) / 1.4; b = hi * 1.4;
      X = (v) => x0 + (Math.log(v) - Math.log(a)) / (Math.log(b) - Math.log(a)) * panelW;
      ticks = logTicks(a, b);
    } else {
      const pad = (hi - lo) * 0.05 || 0.002; a = lo - pad; b = hi + pad;
      X = (v) => x0 + (v - a) / (b - a) * panelW;
      ticks = niceTicks(a, b);
    }
    s += svgText(x0 + panelW / 2, 16, p.title, 'text-anchor="middle" font-size="13"');
    s += `<rect x="${x0}" y="${top}" width="${panelW}" height="${rowH * t.length}" fill="none" stroke="#8c959f"/>`;
    for (const v of ticks) {
      const x = X(v);
      s += line(x, top, x, top + rowH * t.length, 'stroke="#d1d9e0"');
      s += svgText(x, top + rowH * t.length + 16, String(v), 'text-anchor="middle" font-size="11" fill="#59636e"');
    }
    if (!p.log && a < 0.5 && b > 0.5) s += line(X(0.5), top, X(0.5), top + rowH * t.length, 'stroke="#7F7F7F" stroke-dasharray="5,4"');
    t.forEach((r, i) => {
      const y = top + rowH * (i + 0.5), color = CONF.label_colors[r.group] || "#0072B2";
      s += line(X(r[`${p.key} CI low`]), y, X(r[`${p.key} CI high`]), y, `stroke="${color}" stroke-width="9" stroke-opacity="0.6"`);
      const m = X(r[`${p.key} median`]);
      s += line(m, y - 9, m, y + 9, 'stroke="black" stroke-width="2.4"');
      const pv = r[`${p.key} p`];
      s += svgText(x0 + panelW + 6, y + 4, `p=${pyFixed(pv, 3)}` + (p.key === "rank" ? ` · ${r.evidence}` : ""),
        `font-size="11"${pv < 0.05 ? ' font-weight="bold"' : ""}`);
    });
  });
  s += svgText(W - 4, H - 6, "black = group median · bar = its 95% CI (order statistics) · p: permutation test, group median vs rest of the prompt (bold < 0.05)",
    'text-anchor="end" font-size="11" fill="#59636e"');
  return s + "</svg>";
}

function rfPlot(CURVES, model, prompt, k) {
  const design = DESIGN[prompt], labels = META.rf_designs[design], [first, second] = CONF.rf_colors[design];
  const curves = CURVES.filter((r) => r.model === model && r.prompt === prompt);
  const attrs = Object.keys(labels.names), metrics = Object.keys(META.rf_metrics);
  const kMax = Math.max(...curves.map((r) => r.k));
  const pw = 255, ph = 165, ml = 70, mt = 30, gx = 34, gy = 36, legendW = 250;
  const W = ml + 4 * pw + 3 * gx + legendW, H = mt + attrs.length * ph + (attrs.length - 1) * gy + 70;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="Metrics and shares as a function of k">`;
  const panel = (x0, y0, series, {band, nullMean, hline, title, ylabel, last}) => {
    const ys = series.flatMap((c) => c.pts.map((p) => p[1])).concat(band ? band.flatMap((p) => [p[1], p[2]]) : [])
      .concat(nullMean ? nullMean.map((p) => p[1]) : []).concat(hline !== undefined ? [hline] : []).filter(isNum);
    let lo = Math.min(...ys), hi = Math.max(...ys); const pad = (hi - lo) * 0.08 || 0.05; lo -= pad; hi += pad;
    const X = (v) => x0 + (v - 1) / (kMax - 1) * pw, Y = (v) => y0 + ph - (v - lo) / (hi - lo) * ph;
    let o = `<rect x="${x0}" y="${y0}" width="${pw}" height="${ph}" fill="none" stroke="#8c959f"/>`;
    for (const v of niceTicks(lo, hi, 4)) {
      o += line(x0, Y(v), x0 + pw, Y(v), 'stroke="#eaeef2"');
      o += svgText(x0 - 5, Y(v) + 4, String(v), 'text-anchor="end" font-size="10" fill="#59636e"');
    }
    for (let kk = 4; kk <= kMax; kk += 4) {
      o += line(X(kk), y0, X(kk), y0 + ph, 'stroke="#eaeef2"');
      if (last) o += svgText(X(kk), y0 + ph + 14, String(kk), 'text-anchor="middle" font-size="10" fill="#59636e"');
    }
    if (band) o += `<polygon points="${band.map((p) => `${X(p[0]).toFixed(1)},${Y(p[2]).toFixed(1)}`).join(" ")} ${band.slice().reverse().map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="#7F7F7F" fill-opacity="0.22"/>`;
    if (nullMean) o += `<polyline points="${nullMean.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="#7F7F7F" stroke-dasharray="5,4"/>`;
    if (hline !== undefined) o += line(x0, Y(hline), x0 + pw, Y(hline), 'stroke="#7F7F7F" stroke-dasharray="2,3"');
    o += line(X(k), y0, X(k), y0 + ph, 'stroke="#D55E00" stroke-dasharray="2,3"');
    for (const c of series) {
      o += `<polyline points="${c.pts.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ")}" fill="none" stroke="${c.color}" stroke-width="${c.width || 1.8}"${c.dash ? ' stroke-dasharray="6,4"' : ""}/>`;
      if (c.dots) for (const p of c.pts.filter((q) => q[0] % 4 === 0)) o += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.6" fill="${c.color}"/>`;
    }
    if (title) o += svgText(x0 + pw / 2, y0 - 9, title, 'text-anchor="middle" font-size="12"');
    if (ylabel) o += svgText(x0 - 50, y0 + ph / 2, ylabel, `text-anchor="middle" font-size="11" transform="rotate(-90 ${x0 - 50} ${y0 + ph / 2})"`);
    if (last) o += svgText(x0 + pw / 2, y0 + ph + 30, "k", 'text-anchor="middle" font-size="11"');
    return o;
  };
  attrs.forEach((attr, i) => {
    const name = labels.names[attr], y0 = mt + i * (ph + gy), last = i === attrs.length - 1;
    const sel = (series) => curves.filter((r) => r.attribute === name && r.series === series).sort((a, b) => a.k - b.k);
    metrics.forEach((m, j) => {
      const c = sel(m);
      s += panel(ml + j * (pw + gx), y0, [{pts: c.map((r) => [r.k, r.observed]), color: "black", width: 1.6, dots: true}],
        {band: c.map((r) => [r.k, r["null low"], r["null high"]]), nullMean: c.map((r) => [r.k, r["null mean"]]),
         title: i === 0 ? `${m}@k` : "", ylabel: j === 0 ? name : "", last});
    });
    const groups = labels.groups[attr];
    const series = groups.map((g, gi) => {
      const c = sel(`share: ${g}`);
      const color = attr === "b" ? second[gi] : first[attr === "ab" ? [0, 0, 1, 1][gi] : gi];
      return {pts: c.map((r) => [r.k, r.observed]), color, dash: attr === "ab" && gi % 2 === 1, label: g, band: c};
    });
    const band = series[series.length - 1].band;
    const x0 = ml + 3 * (pw + gx);
    s += panel(x0, y0, series, {band: band.map((r) => [r.k, r["null low"], r["null high"]]), hline: 1 / groups.length,
      title: i === 0 ? "share of top-k" : "", last});
    series.forEach((c, gi) => {
      const ly = y0 + 14 + gi * 17;
      s += line(x0 + pw + 12, ly - 4, x0 + pw + 34, ly - 4, `stroke="${c.color}" stroke-width="2.4"${c.dash ? ' stroke-dasharray="6,4"' : ""}`);
      s += svgText(x0 + pw + 40, ly, c.label, 'font-size="11"');
    });
  });
  s += svgText(W - 4, H - 8, "black = observed (mean over balanced draws) · dashed grey = random-order expectation, band = its 95% range " +
    `(permutation null) · dotted grey = balanced share · dotted vermillion = selected k · 2×2 shares: dashed = ${labels.groups.b[1]}`,
    'text-anchor="end" font-size="11" fill="#59636e"');
  return s + "</svg>";
}

// ---- Page -----------------------------------------------------------------------------------------

function guard(fn) {
  return () => fn().catch((e) => {
    console.error(e);
    $("#summary").innerHTML = `<span class="error">Could not load the data: ${esc(e.message)}</span>`;
  });
}

async function init() {
  [META, CONF] = await Promise.all([load("data/meta.json"), load("static/config.json")]);
  MODELS = META.models.map((m) => m.key);
  PROMPTS = META.prompts.map((p) => p.prompt);
  DESIGN = Object.fromEntries(META.prompts.map((p) => [p.prompt, p.design]));
  RF_PROMPTS = META.prompts.filter((p) => p.ranking_fairness).map((p) => p.prompt);
  KS = META.ks; MIN_N = META.min_n; RFC = META.rf_constants; DIPC = META.dip_constants;

  $("#summary").textContent = `${MODELS.length} models · ${PROMPTS.length} prompts · ${fmtInt(META.images)} annotated images ` +
    `(not public) · release ${META.release}. Aggregate results only; see About.`;
  const T = texts();
  document.querySelectorAll("[data-md]").forEach((el) => { el.innerHTML = md(T[el.dataset.md]); });
  $("#rf-prompt-label").textContent = `Prompt (smallest 2×2 cell ≥ ${RFC.min_cell})`;

  options($("#model"), MODELS, MODELS[0]);
  options($("#lb-config"), Object.keys(META.leaderboard));
  options($("#cmp-a"), MODELS, MODELS[0]);
  options($("#cmp-b"), MODELS, MODELS[MODELS.length - 1]);
  options($("#dip-prompt"), PROMPTS, PROMPTS[0]);
  options($("#dip-grouping"), META.groupings, "Gender");
  options($("#dip-sort"), Object.keys(SORT_KEYS));
  options($("#rf-prompt"), RF_PROMPTS, RF_PROMPTS[0]);
  $("#ev-groupings").innerHTML = META.groupings.map((g) =>
    `<label><input type="checkbox" value="${esc(g)}"${META.default_groupings.includes(g) ? " checked" : ""}> ${esc(g)}</label>`).join("");
  ["lb-k", "cmp-k", "rf-k"].forEach(kSlider);

  const on = (ids, fn, ev = "change") => ids.forEach((id) => $(`#${id}`).addEventListener(ev, guard(fn)));
  on(["lb-config", "lb-k"], leaderboard);
  on(["cmp-a", "cmp-b", "cmp-k"], compareModels);
  on(["model", "dip-prompt", "dip-grouping", "dip-sort"], groupMetrics);
  on(["model", "ev-uncertain"], evidenceAll);
  $("#ev-groupings").addEventListener("change", guard(evidenceAll));
  on(["model", "rf-prompt", "rf-k"], rankingFairness);
  on(["model", "rf-k"], rankingFairnessAll);

  // Tabs (the URL hash keeps the open tab).
  const tabs = [...document.querySelectorAll("nav button")];
  const show = (name) => {
    tabs.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll("main > section").forEach((s) => { s.hidden = s.id !== `tab-${name}`; });
  };
  tabs.forEach((b) => b.addEventListener("click", () => { show(b.dataset.tab); history.replaceState(null, "", `#${b.dataset.tab}`); }));
  const start = location.hash.slice(1);
  if (tabs.some((b) => b.dataset.tab === start)) show(start);

  await Promise.all([leaderboard(), compareModels(), groupMetrics(), evidenceAll(), rankingFairness(), rankingFairnessAll()]);
}

guard(init)();
