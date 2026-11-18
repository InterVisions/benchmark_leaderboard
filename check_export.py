"""Leak check of the published data: fails unless data/ holds only aggregate numbers.

    python check_export.py [DIR]      (default: this repository)

Checks:
* the repository holds only the files of REPO_FILES, data/ and evaluation/ (all files, not just
  the tracked ones; caches such as __pycache__ are skipped);
* only the expected files in data/ and evaluation/, in each CSV only the expected columns, and in
  meta.json only the expected keys;
* the texts (README.md, LICENSE, docs/, evaluation/, and the static page: index.html and static/)
  have no URL except those of TEXT_ALLOW, so the page loads nothing from other sites, and none of
  the patterns below;
* no URL, e-mail address, image file name or annotation-tool image id in any text;
* no annotator labels;
* every published group has n >= min_n (meta.json);
* no table has a row per image: every CSV row is keyed by model/prompt/group, k or curve point;
* meta.json's annotations_sha256 (the SHA-256 of the private annotation file) is a SHA-256 hex
  digest, and README.md, when present, states the same digest in its integrity block.
The app runs this check at startup and refuses to serve data that fails it.
"""
import csv
import json
import re
import sys
from pathlib import Path

SCHEMA = {
    "groups.csv": ["model", "prompt", "grouping", "group", "n", "evidence", "direction",
                   "other models", "AUC median", "AUC CI low", "AUC CI high", "AUC Δ median",
                   "AUC p", "AUC mean", "rank median", "rank CI low", "rank CI high",
                   "rank Δ median", "rank p", "rank p Holm", "rank mean", "R@1", "R@5", "R@10"],
    "spread.csv": ["model", "prompt", "grouping", "assessed groups", "spread", "null mean", "p"],
    "utility.csv": ["model", "prompt", "images", "COCO median rank", "COCO R@1", "COCO R@5",
                    "COCO R@10", "AUC median"],
    "rf_metrics.csv": ["model", "prompt", "k", "attribute", "metric", "observed",
                       "random-order expectation", "observed − expected", "p (one-sided)",
                       "per-draw 2.5%", "per-draw 97.5%", "null 95% range low",
                       "null 95% range high"],
    "rf_shares.csv": ["model", "prompt", "k", "attribute", "group", "balanced share",
                      "share − balanced", "p (two-sided)", "share"],
    "rf_curves.csv": ["model", "prompt", "attribute", "series", "k", "observed", "null mean",
                      "null low", "null high"],
    "rf_cells.csv": ["prompt", "design", "cell", "n"],
    "paired.csv": ["model A", "model B", "prompt", "k", "attribute", "metric", "A − B", "p"],
}
META_KEYS = {"release", "annotations_sha256", "images", "min_n", "ks", "models", "prompts",
             "groupings", "default_groupings", "leaderboard", "rf_designs", "rf_metrics",
             "rf_constants", "dip_constants"}
SHA256 = re.compile(r"[0-9a-f]{64}")
README_BLOCK = re.compile(r"<!-- annotations-sha256 -->\n(.*?)\n<!-- /annotations-sha256 -->", re.S)
EVAL_CODE = {"dataset.py", "metrics.py", "ranking_fairness.py", "codebook.md"}
# Every other file the repository may hold (paths relative to its root).
REPO_FILES = {"README.md", "LICENSE", "app.py", "check_export.py", "requirements.txt", "index.html",
              "static/leaderboard.js", "static/config.json", ".gitignore", "eu-funded.png",
              "docs/datasheet.md"}
SKIP_DIRS = {".git", "__pycache__", ".gradio", ".venv", "flagged"}
# Texts scanned for URLs and the patterns below: what is published as prose, and the static page,
# which must load nothing from other sites.
TEXTS = ["README.md", "LICENSE", "docs/datasheet.md", "index.html", "static/leaderboard.js",
         "static/config.json"]
# Links the texts may contain, and the emblem's file name.
TEXT_ALLOW = re.compile(r"https?://(github\.com/InterVisions/|huggingface\.co/|"
                        r"creativecommons\.org/|opensource\.org/|commission\.europa\.eu/|127\.0\.0\.1:)\S*|"
                        r"eu-funded\.png")
FORBIDDEN = [
    (re.compile(r"https?://|www\.", re.I), "URL"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "e-mail address"),
    (re.compile(r"\.(jpe?g|png|gif|webp|bmp|tiff?)\b", re.I), "image file name"),
    (re.compile(r"\b(task|ann)\d+_\d{6,}"), "annotation-tool image id"),
    (re.compile(r"annotator \d+", re.I), "annotator label"),
]
# Text that legitimately matches a pattern in the published evaluation code (docstrings, paths).
CODE_ALLOW = re.compile(r"https?://images\.cocodataset\.org/\S*|\.npz|\.csv")


def fail(msg):
    sys.exit(f"check_export: {msg}")


def scan(text, where, allow=None):
    for line in text.splitlines():
        probe = allow.sub("", line) if allow else line
        for pattern, what in FORBIDDEN:
            if pattern.search(probe):
                fail(f"{what} in {where}: {line.strip()[:120]}")


def repo_files(root):
    out = set()
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if p.is_file() and not SKIP_DIRS & set(rel.parts):
            out.add(rel.as_posix())
    return out


def main(root):
    data, code = root / "data", root / "evaluation"
    extra = {f for f in repo_files(root) if f.split("/")[0] not in ("data", "evaluation")}
    if extra - REPO_FILES:
        fail(f"unexpected files: {sorted(extra - REPO_FILES)} (add them to REPO_FILES if public)")
    for name in TEXTS:
        if (root / name).is_file():
            scan((root / name).read_text(), name, TEXT_ALLOW)
    emblem = root / "eu-funded.png"
    if emblem.is_file() and (emblem.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
                             or emblem.stat().st_size > 100_000):
        fail("eu-funded.png is not the small PNG emblem")
    files = {p.name for p in data.iterdir()}
    if files != set(SCHEMA) | {"meta.json"}:
        fail(f"unexpected files in data/: {sorted(files ^ (set(SCHEMA) | {'meta.json'}))}")
    meta = json.loads((data / "meta.json").read_text())
    if set(meta) != META_KEYS:
        fail(f"meta.json: keys {sorted(set(meta) ^ META_KEYS)} differ from the allow-list")
    digest = meta["annotations_sha256"]
    if not (isinstance(digest, str) and SHA256.fullmatch(digest)):
        fail("meta.json: annotations_sha256 is not a SHA-256 hex digest")
    scan(json.dumps(meta, ensure_ascii=False), "meta.json")
    readme = root / "README.md"
    if readme.is_file():
        block = README_BLOCK.search(readme.read_text())
        if not block or SHA256.findall(block.group(1)) != [digest]:
            fail("README.md: the integrity block does not state meta.json's annotations_sha256")
    min_n = int(meta["min_n"])
    for name, cols in SCHEMA.items():
        with open(data / name, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header != cols:
                fail(f"{name}: columns {header} differ from the schema")
            for row in reader:
                if len(row) != len(cols):
                    fail(f"{name}: malformed row {row[:4]}")
                scan(" | ".join(row), name)
                if name == "groups.csv" and int(row[cols.index("n")]) < min_n:
                    fail(f"groups.csv: a group with n < {min_n} ({row[:4]})")
    if code.exists():
        names = {p.name for p in code.iterdir() if p.is_file()}
        if not names <= EVAL_CODE:
            fail(f"unexpected files in evaluation/: {sorted(names - EVAL_CODE)}")
        for p in code.iterdir():
            if p.is_file():
                allow = TEXT_ALLOW if p.suffix == ".md" else CODE_ALLOW
                scan(p.read_text(), f"evaluation/{p.name}", allow)
    print(f"check_export: {root} OK ({len(SCHEMA)} tables, groups with n >= {min_n}, "
          f"annotations sha256 {digest})")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent))
