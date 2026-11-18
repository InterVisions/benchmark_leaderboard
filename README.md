---
title: InterVisions Leaderboard
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Fairness of text-to-image retrieval models
---

# InterVisions leaderboard

## What the benchmark measures

How text-to-image retrieval models (CLIP and related vision–language models) treat people of
different perceived gender, age and skin tone. Each model retrieves images for natural-language
prompts about people in public spaces, and two analyses compare the groups of people depicted:

- **Difference in performance**: per-group retrieval utility of the relevant images (in-dataset AUC
  and rank among 5,000 COCO val2017 distractors), with permutation tests and graded evidence.
- **Ranking fairness**: NDKL, MaxSkew and normalized entropy of the top-k on balanced 2×2 pools
  (perceived gender × skin tone, or the composition of couples), against a permutation null.

The dataset, its prompts and the limits of its perceived attributes are described in
[`docs/datasheet.md`](docs/datasheet.md); the codes are in
[`evaluation/codebook.md`](evaluation/codebook.md).

## What is not released, and why

The images, their URLs, the per-image annotations, the annotators' notes and the embeddings are
not released. The images show identifiable people, including minors, and the annotations record
perceived characteristics such as skin tone, so publishing them would expose personal data of
people who were never asked; the images also belong to their authors. This repository therefore
publishes only aggregate numbers over groups of at least 10 images, with the evaluation code and
the hash of the annotation file.

## What is published

Only aggregate numbers, exported from the private benchmark: per model, prompt and group, the
summaries of the two analyses. No number refers to a single image. Groups with fewer than 10
images are neither shown nor assessed. `check_export.py` verifies this: only the expected files,
columns and keys, no URLs, image file names, annotator labels or small groups, and a static page
that loads nothing from other sites. It runs before every release is published, and the Gradio
version of the app also runs it at startup.

```
index.html          the leaderboard: a static page (no server; reads data/ in the browser)
static/             its script (leaderboard.js) and configuration (config.json: group colors,
                    evidence rules)
app.py              the same leaderboard as a Gradio app, for running it locally
data/               exported tables (CSV) and meta.json
evaluation/         evaluation code (groupings, metrics, evidence rules, ranking fairness design)
                    and the annotation codebook
docs/datasheet.md   the dataset in aggregate: prompts, counts, perceived attributes and their limits
check_export.py     leak check of the whole repository, and of the published hash
```

## Release and integrity

The annotation file is private, but its SHA-256 is published with every release (also in
`data/meta.json`), so anyone given access to the data later can verify that it is the file these
numbers were computed from.

<!-- annotations-sha256 -->
Release 2026-11-16: SHA-256 of the private annotation file (`annotations.csv`, 1,203 images) the published numbers were computed from:

```
ece447fab4a42bc9113c64db01330a1a18eb76fce840c5f6895fb892a48e66aa
```
<!-- /annotations-sha256 -->

## Run locally

The static page needs only a local web server (browsers do not read `data/` from a file:// page):

```bash
python -m http.server 8000      # then open http://127.0.0.1:8000/
```

The Gradio version shows the same views and numbers:

```bash
pip install -r requirements.txt
python app.py                   # http://127.0.0.1:7860
```

## Submit a model

Submissions are evaluated by the maintainers on the private data, through a manual queue. Open a
discussion in the Space (or an issue in this repository) with:

- the model name and a public checkpoint: an `open_clip` architecture and pretrained tag, or a
  Hugging Face model id;
- how to compute its image and text embeddings, if it is not an `open_clip` model;
- a link to its paper or model card.

The weights must be in **safetensors** format, or a PyTorch state dict that loads with
`torch.load(weights_only=True)`. Pickled models and TorchScript archives are refused, because
loading them can run arbitrary code. The maintainers compute the results privately and add the
model's aggregate numbers to the leaderboard; nothing else is sent back.

## Related

The dataset was collected and annotated with the InterVisions annotation tool:
<https://github.com/InterVisions/intervisions_annotation>.

## License

- **Code** (`index.html`, `static/leaderboard.js`, `app.py`, `check_export.py`, `evaluation/*.py`):
  MIT, see [`LICENSE`](LICENSE).
- **Data** (the tables in `data/`, `docs/datasheet.md` and `evaluation/codebook.md`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Funding acknowledgement

![Co-funded by the European Union](eu-funded.png)

This work is part of the InterVisions project (101214711 — CERV-2024-CHAR-LITI). Funded by the
European Union. Views and opinions expressed are however those of the author(s) only and do not
necessarily reflect those of the European Union or the European Education and Culture Executive
Agency (EACEA). Neither the European Union nor EACEA can be held responsible for them.
