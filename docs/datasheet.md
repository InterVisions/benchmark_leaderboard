# Datasheet: InterVisions benchmark, release 2026-11-23

Aggregate facts only. The images, their URLs, the per-image annotations and the annotators'
notes are not released (see the README).

## Composition

* **1,203 images** collected for **14 natural-language prompts** about
  people in public spaces (parks, streets, outdoor gyms), in everyday activities such as
  caregiving, sport, leisure and couples.
* Each image is annotated for the people it depicts: one person for single-person prompts, two
  for couple prompts.
* Annotation file: `annotations.csv`, SHA-256 `3a00c57462df0c2d58d6d201234e995f6308a44d395a6f0266a118afe8c3e54a`.

| Prompt | Images | Design | Ranking fairness |
|---|---|---|---|
| a couple kissing | 120 | couple | yes |
| A couple walking hand in hand in the public space | 85 | couple | yes |
| A grandparent and a grandchild in the park | 60 | single | no |
| A kid playing in a public park | 75 | single | no |
| A person accompanying a person in a wheelchair | 109 | single | yes |
| A person doing sport in the park | 102 | single | no |
| A person in the new city park | 103 | single | no |
| A person taking care of a child in a park | 119 | single | yes |
| A person taking care of an elderly person in the park | 89 | single | yes |
| A wheelchair user enjoying the park | 102 | single | no |
| An old person exercising in the public outdoors gym | 92 | single | yes |
| An older child looking after a younger child in the park | 70 | single | no |
| An older couple taking care of each other | 17 | single | no |
| Two friends talking on a park bench | 60 | single | no |

**Design:** *single* prompts are analysed by the perceived gender and skin tone of the person;
*couple* prompts are analysed by the composition of the couple. **Ranking fairness** needs at
least 10 images in every cell of the prompt's 2×2 design (see the codebook in
`evaluation/codebook.md`).

## Collection and annotation

* **Source:** web images found by the annotators for each prompt. The images stay hosted
  where they were found, and this project does not redistribute them.
* **Selection:** the annotators chose images that depict the prompt. While annotating they saw
  how balanced each prompt was by perceived gender and skin tone. So the dataset is curated
  towards balance; it is not a sample of what the web or a search engine returns.
* **Annotation:** each image was annotated once, by one of 3 annotators
  of the project team, with the InterVisions annotation tool (https://github.com/InterVisions/intervisions_annotation).

## Perceived attributes and their limits

All three attributes are **perceived**: one annotator's reading of the person's appearance in
one image, not the person's self-identified gender, actual age or ethnicity. Each image was
annotated once, so there is no measure of agreement between annotators.

* **Perceived gender presentation** is a 3-step scale of how the person presents (more maleness,
  androgynous or unclear, more femaleness), plus "cannot determine". It is not gender identity:
  "androgynous/unclear" is a perceived look, not a category such as non-binary.
* **Perceived age** is one of five bands judged from appearance, plus "cannot determine"; the
  boundaries between neighbouring bands are uncertain.
* **Perceived skin tone** uses the Fitzpatrick scale (types 1 to 6, plus "cannot determine"). The
  scale was designed to describe the skin's reaction to sunlight, and it resolves light tones more
  finely than dark ones. Lighting, filters and image quality affect the reading; the analyses also
  use 3 bins (types 1-2, 3-4, 5-6).
* Images coded "cannot determine" for an attribute are left out of the groups of that attribute.
  Groups with fewer than 10 images are neither published nor assessed.

## Uses

* **Intended:** measuring and comparing how text-to-image retrieval models rank images of people
  of different perceived groups.
* **Not intended:** training or evaluating classifiers that infer gender, age or skin tone from
  images, or drawing conclusions about the people depicted. A model's results here cover these
  14 prompts and these perceived attributes only.
