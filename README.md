---
title: InterVisions Leaderboard
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
short_description: Fairness of text-to-image retrieval models
---

# InterVisions leaderboard

How text-to-image retrieval models (CLIP and related vision–language models) treat people of
different perceived gender, age and skin tone. Each model retrieves images for natural-language
prompts about people in public spaces, and two analyses compare the groups of people depicted:

- **Difference in performance**: per-group retrieval utility of the relevant images (in-dataset AUC
  and rank among 5,000 COCO val2017 distractors), with permutation tests and graded evidence.
- **Ranking fairness**: NDKL, MaxSkew and normalized entropy of the top-k on balanced 2×2 pools
  (perceived gender × skin tone, or the composition of couples), against a permutation null.

## What is published

Only aggregate numbers, exported from the private benchmark: per model, prompt and group, the
summaries of the two analyses. The images, their URLs, the annotations and the embeddings are not
public, and no number refers to a single image. Groups with fewer than 10 images are neither shown
nor assessed. `check_export.py` verifies this (expected files and columns only; no URLs, image file
names, annotator labels or small groups); the app runs it at startup and refuses to serve data that
fails it.

```
app.py              Gradio app (reads data/)
data/               exported tables (CSV) and meta.json
evaluation/         evaluation code: groupings, metrics, evidence rules, ranking fairness design
check_export.py     leak check of data/ and evaluation/, and of the published hash
```

## Integrity

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

```bash
pip install -r requirements.txt
python app.py          # http://127.0.0.1:7860
```

## Submit a model

Submissions are evaluated by the maintainers on the private data, through a manual queue: open a
discussion in the Space (or an issue in this repository) with the model name, a public checkpoint
(an `open_clip` architecture and pretrained tag, or a Hugging Face model id), how to compute its
image and text embeddings if it is not an `open_clip` model, and a link to its paper or model card.
