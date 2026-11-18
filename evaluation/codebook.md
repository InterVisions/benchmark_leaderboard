# Annotation codebook

The codes of the perceived attributes, as used by the evaluation code in this directory
(`dataset.py`). The annotations themselves are private; see `docs/datasheet.md` for what the
dataset contains and for the limits of these attributes.

## Perceived gender presentation (`perceived_gender`)

| Code | Label |
|---|---|
| `0` | more maleness |
| `1` | androgynous/unclear |
| `2` | more femaleness |
| `-1` | cannot determine |

## Perceived age (`perceived_age`)

| Value | Label in the analyses |
|---|---|
| Child (0-12) | child |
| Adolescent (13-17) | adolescent |
| Young adult (18-30) | young adult |
| Middle-aged (31-60) | middle-aged |
| Older adult (60+) | older adult |
| Cannot determine | (not in any age group) |

## Perceived skin tone (`perceived_skin_tone`, Fitzpatrick type)

| Code | Label | 3-bin label |
|---|---|---|
| `1` | FST 1 | light (FST 1-2) |
| `2` | FST 2 | light (FST 1-2) |
| `3` | FST 3 | medium (FST 3-4) |
| `4` | FST 4 | medium (FST 3-4) |
| `5` | FST 5 | dark (FST 5-6) |
| `6` | FST 6 | dark (FST 5-6) |
| `0` | cannot determine | (not in any skin tone group) |

## Couples

Prompts about couples annotate the same three attributes for a second person (`p2_` columns),
and the couple groupings describe the pair: same or different perceived gender, and the skin tone
bins of the two people.

## Groupings used in the analyses

* Gender
* Skin tone (6 levels)
* Skin tone (3 bins)
* Age
* Gender × skin tone (3 bins)
* Gender × age
* Skin tone (3 bins) × age
* Gender × skin tone (3 bins) × age
* Couples: gender composition
* Couples: skin tone composition (3 bins)
* Couples: gender × skin composition (2×2)
* Number of people
