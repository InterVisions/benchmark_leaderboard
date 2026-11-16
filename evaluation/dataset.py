"""InterVisions annotations, demographic groupings and the COCO distractor gallery.

Images are not stored: the app shows them from their public URLs (`image_url`).
"""
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "annotations.csv"
COCO_CSV = DATA_DIR / "coco_val2017.csv"
# images.cocodataset.org only serves http; behind https, point this to a proxy of it.
COCO_URL = os.environ.get("COCO_IMAGE_URL", "http://images.cocodataset.org/val2017/{}")

# Codebook and colors of the annotation tool.
# Gender is perceived gender presentation; 1 is a perceived look (androgynous or ambiguous),
# not a gender identity such as non-binary.
GENDER_LABELS = {"0": "more maleness", "1": "androgynous/unclear", "2": "more femaleness",
                 "-1": "cannot determine"}
GENDER_COLORS = {"0": "#4A8BC2", "1": "#A0A0A0", "2": "#C77DBA", "-1": "#BBBBBB"}
# Skin tone is the Fitzpatrick scale, 1 (lightest) to 6 (darkest); 0 = cannot determine.
SKIN_LABELS = {str(i): f"FST {i}" for i in range(1, 7)}
FST_COLORS = {"1": "#c6b49d", "2": "#bda389", "3": "#af9478", "4": "#a48367", "5": "#886951",
              "6": "#664e41"}

SHOW_COLS = ["term", "num_people", "perceived_gender", "perceived_age", "perceived_skin_tone",
             "p2_perceived_gender", "p2_perceived_age", "p2_perceived_skin_tone",
             "annotator", "intersectional_notes"]


def load_annotations() -> pd.DataFrame:
    """One row per annotated image (`image_id` keys the precomputed embeddings)."""
    return pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)


def terms(df: pd.DataFrame) -> list[str]:
    return sorted(df.term.unique(), key=str.lower)


def _person(gender, age, skin):
    g = GENDER_LABELS.get(gender, gender) or "?"
    s = SKIN_LABELS.get(skin, "FST ?")
    return f"{g}, {age.split(' (')[0] or '?'}, {s}"


def caption(row) -> str:
    text = _person(row.perceived_gender, row.perceived_age, row.perceived_skin_tone)
    if row.annotation_type == "couple":
        text += " | " + _person(row.p2_perceived_gender, row.p2_perceived_age,
                                row.p2_perceived_skin_tone)
    return text


# ---- Demographic groupings ------------------------------------------------------------------
# Each grouping maps the annotation table to one label per image ("" = image not in any group).

AGE_SHORT = {"Child (0-12)": "child", "Adolescent (13-17)": "adolescent",
             "Young adult (18-30)": "young adult", "Middle-aged (31-60)": "middle-aged",
             "Older adult (60+)": "older adult"}
SKIN3 = {"1": "light (FST 1-2)", "2": "light (FST 1-2)", "3": "medium (FST 3-4)",
         "4": "medium (FST 3-4)", "5": "dark (FST 5-6)", "6": "dark (FST 5-6)"}


def _mix(*hexes):
    rgb = [[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes]
    return "#" + "".join(f"{round(sum(c) / len(c)):02x}" for c in zip(*rgb))


# Colors keyed by the group labels used below (3-bin colors average their two FST colors).
LABEL_COLORS = {**{SKIN_LABELS[k]: v for k, v in FST_COLORS.items()},
                **{SKIN3[a]: _mix(FST_COLORS[a], FST_COLORS[b]) for a, b in ("12", "34", "56")},
                **{GENDER_LABELS[k]: v for k, v in GENDER_COLORS.items()},
                "FST 2-6": _mix(*(FST_COLORS[k] for k in "23456"))}


def label_color(label, default="#0072B2"):
    """Color of a group label: its skin-tone component if any, else its gender component."""
    parts = [p for part in label.split(" × ") for p in part.split(" + ")]
    skin = [LABEL_COLORS[p] for p in parts if p.startswith(("FST", "light", "medium", "dark"))]
    gender = [LABEL_COLORS[p] for p in parts if p in GENDER_LABELS.values()]
    colors = skin or gender
    return _mix(*colors) if colors else default


def _gender(s):
    return s.map(lambda g: GENDER_LABELS.get(g, f"gender code {g}") if g != "" else "")


def _age(s):
    return s.map(lambda a: AGE_SHORT.get(a, ""))


def _skin6(s):
    return s.map(lambda v: SKIN_LABELS.get(v, ""))


def _skin3(s):
    return s.map(lambda v: SKIN3.get(v, ""))


def _cross(*cols):
    out = cols[0]
    for c in cols[1:]:
        out = out.combine(c, lambda a, b: f"{a} × {b}" if a and b else "")
    return out


def _pair(df, fn, col):
    """Sorted composition of a couple ("" for non-couple images or missing labels)."""
    a, b = fn(df[f"perceived_{col}"]), fn(df[f"p2_perceived_{col}"])
    pair = a.combine(b, lambda x, y: " + ".join(sorted([x, y])) if x and y else "")
    return pair.where(df.annotation_type == "couple", "")


def _couple_2x2(df):
    """Gender composition × skin tone composition with the split of the ranking fairness design
    (different / same gender × both FST 1-2 / other); "" when either person lacks a code."""
    fst = lambda col: pd.to_numeric(df[col].where(df[col].isin(list("123456"))), errors="coerce")
    g1, g2 = df.perceived_gender, df.p2_perceived_gender
    s1, s2 = fst("perceived_skin_tone"), fst("p2_perceived_skin_tone")
    ok = (df.annotation_type == "couple") & g1.isin(["0", "2"]) & g2.isin(["0", "2"]) \
        & s1.notna() & s2.notna()
    gender = (g1 == g2).map({True: "same gender", False: "different gender"})
    skin = ((s1 <= 2) & (s2 <= 2)).map({True: "both FST 1-2", False: "other"})
    return (gender + " × " + skin).where(ok, "")


GROUPINGS = {
    "Gender": lambda df: _gender(df.perceived_gender),
    "Skin tone (6 levels)": lambda df: _skin6(df.perceived_skin_tone),
    "Skin tone (3 bins)": lambda df: _skin3(df.perceived_skin_tone),
    "Age": lambda df: _age(df.perceived_age),
    "Gender × skin tone (3 bins)": lambda df: _cross(_gender(df.perceived_gender),
                                                     _skin3(df.perceived_skin_tone)),
    "Gender × age": lambda df: _cross(_gender(df.perceived_gender), _age(df.perceived_age)),
    "Skin tone (3 bins) × age": lambda df: _cross(_skin3(df.perceived_skin_tone),
                                                  _age(df.perceived_age)),
    "Gender × skin tone (3 bins) × age": lambda df: _cross(_gender(df.perceived_gender),
                                                           _skin3(df.perceived_skin_tone),
                                                           _age(df.perceived_age)),
    "Couples: gender composition": lambda df: _pair(df, _gender, "gender"),
    "Couples: skin tone composition (3 bins)": lambda df: _pair(df, _skin3, "skin_tone"),
    "Couples: gender × skin composition (2×2)": _couple_2x2,
    "Number of people": lambda df: df.num_people.map(lambda n: f"{n} people" if n else ""),
}


# ---- COCO distractor gallery ----------------------------------------------------------------

def load_coco() -> pd.DataFrame:
    """COCO val2017 images in id order, with one caption each and their public URL."""
    coco = pd.read_csv(COCO_CSV, dtype={"file_name": str, "caption": str}, keep_default_na=False)
    coco["url"] = coco.file_name.map(COCO_URL.format)
    return coco
