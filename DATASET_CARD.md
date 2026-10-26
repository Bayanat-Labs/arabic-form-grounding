# Arabic Form Grounding v0.1.0

Locate a field by its Arabic section, label or spatial relationship, return its
value and complete field rectangle, and abstain when the request has no single
supported answer. These are **original Bayanat-authored synthetic fixtures**.
They contain no real customer records or source document scans.

[Explorer](https://huggingface.co/spaces/Bayanat-Labs/arabic-form-grounding) ·
[Code and scorer](https://github.com/Bayanat-Labs/arabic-form-grounding/tree/v0.1.0) ·
[Research page](https://www.bayanatlabs.com/open/arabic-form-grounding)

The contribution is a narrowly specified, reproducible Arabic field-association
task: repeated field labels under different visible section headings, exact
rendered semantic areas, layout-family-held-out splits, matched digit and
heading-change pairs, and explicit missing/ambiguous cases. It complements
Arabic OCR resources; it does not establish a new general document-understanding
task, natural-form accuracy or human-annotated gold.

| Counted unit | Count |
| --- | ---: |
| Distinct base form contexts | 864 |
| Rendered form images | 1,152 |
| Base image-question contexts before matched views | 6,912 |
| Image-question rows, including matched views | 9,216 |
| Matched question pairs | 2,304 |
| Layout geometry families | 24 |
| Fictional form domains | 6 |

The extra 288 images are 144 Eastern-digit views and 144 section-heading
counterfactuals. They are repeated controlled views, not new independent forms.
Each question has Arabic and controlled English wording in the **same row**;
translations are not counted as additional examples. All gold labels are public
in every split. Images are 1,280 × 1,600 PNGs, embedded as bytes in Parquet.
Distinct contexts still share controlled grammar, schema and layout components;
these counts do not assert statistical independence.

| Split | Layout families | Images | Question rows |
| --- | --- | ---: | ---: |
| Train | 00–15 | 768 | 6,144 |
| Validation | 16–19 | 192 | 1,536 |
| Test | 20–23 | 192 | 1,536 |

All variants and questions sharing a base context stay together. The 24 geometry
families combine six section arrangements and four field arrangements; validation
and test hold out complete section arrangements. The same four field-arrangement
patterns and controlled query grammars recur across splits. Domains occur in all
splits. This is a scoped layout holdout, not evidence of broad out-of-distribution
generalization. Values are fictional; some status/count/date values repeat.

## Load and inspect

```python
import json
from datasets import load_dataset

forms = load_dataset("Bayanat-Labs/arabic-form-grounding", "forms",
                     split="test", revision="v0.1.0")
questions = load_dataset("Bayanat-Labs/arabic-form-grounding", "questions",
                         split="test", revision="v0.1.0")
form = forms[0]
elements = json.loads(form["elements_json"])
q = next(q for q in questions if q["form_id"] == form["form_id"])
print(q["question_ar"], q["gold_status"], q["gold_bbox"])
image = form["image"]
```

The source JSONL files under `data/` preserve full rendering audit records;
the three Parquet configurations provide forms, questions and pairs. Join by
`form_id`; group evaluation uncertainty by `family_id`.

## Tasks and output

Every image has eight controlled requests: two section-scoped label requests,
one nearest-center relation, one extreme-center relation, one unscoped repeated
label, one scoped missing-field control, one checkbox state and one unique form
reference. The global repeated-label request is deliberately ambiguous. Every
base form omits one field in one section, with the same label visible elsewhere.
No empty placeholder area represents that omitted field. These controls are
simple and regular; they do not establish natural ambiguity or missingness rates.

The 9,216 rows contain 6,408 answerable, 1,800 ambiguous and 1,008 missing cases.
For a section-heading counterfactual, the field contents and geometry remain
fixed while two visible headings swap; the original question is unchanged.
Some answers move to a different field or change from missing to answerable.
Other requests deliberately remain invariant. The pair file labels the measured
relation; 432 heading pairs change target/status and 720 remain invariant.
The 1,152 digit pairs preserve target, status and canonical numeric text.

Return JSON, one record per question:

```json
{"question_id":"...","status":"answerable","text":"59","bbox":[0.5203125,0.3425,0.9515625,0.495]}
```

`status` is `answerable`, `missing` or `ambiguous`. For either abstention,
`text` and `bbox` must be null. An optional `field_id` can be returned in the
oracle-layout track, where element IDs are provided; if supplied, it must match.
Image-only methods can omit it. Opaque IDs never determine a query answer.
Checkbox values are Arabic `محدد` or `غير محدد`.

`bbox` is normalized `[left, top, right, bottom]` relative to the full PNG.
It encloses the **complete semantic field rectangle, including label and value**,
or a checkbox's label-and-square rectangle. It is not a tight text, word, token,
line or glyph box. Spatial relations compare these field-area centers, use
Euclidean distance, and treat exact ties as ambiguous. Section membership comes
from containment within the visible bordered section. See [SCHEMA.md](SCHEMA.md).

Keep `gold_*`, `candidate_field_ids`, `query_spec_json`, variants and generator
metadata out of model inputs. [code/contract.py](code/contract.py) supplies safe
image and oracle-layout input builders. The latter explicitly gives exact text,
field areas and checkbox states; it is not an OCR result.

## Validation and measured baselines

The released-question validator reparses Arabic text and resolves visible text
and geometry without importing the generator or reading `query_spec`. It checks
all 9,216 gold references, 1,152 decoded unique image hashes, family/layout split
separation, 2,304 paired relations and 39,168 contained rendering ink regions.
The renderer checks every drawn codepoint against the actual selected font cmap,
requires at least 18px text (observed minimum 21px), and preserves ISO component
order in Eastern-digit numeric runs. These are programmatic checks, not native
speaker, independent human annotation or glyph-accuracy certification.

| Given-layout rules baseline | Test rows | Joint success |
| --- | ---: | ---: |
| Exact label matching, ignoring sections | 1,536 | 37.5% |
| Exact labels plus visible section containment | 1,536 | 75.0% |

Both baselines receive **oracle semantic text and rectangles**, including
checkbox states. They do not run OCR or a vision model. The second baseline
explicitly lacks nearest/extreme relation handling and returns `missing` for
those requests, so its 25% failure rate is a designed limitation, not evidence
that sophisticated models find the dataset hard. The independent reference
resolver solves the complete controlled task and is not reported as a learned
model baseline. No LLM or vision-model accuracy is claimed.

Joint success requires correct status and, for answerable rows, exact canonical
text plus IoU ≥ 0.5. Canonical text uses NFC, edge trimming and Eastern-to-Western
digit normalization only. Visible-text exact match, mean IoU, IoU ≥ 0.99,
abstention precision/recall, task/layout slices and paired-both-correct rates
are reported separately. Invalid/missing predictions count as wrong; duplicate
or unknown question IDs are rejected. No output repair or LLM judge is used.

Reproduce from a source checkout with Python 3.14, Pillow 12.3.0 with RAQM,
and PyArrow 26.0.0 (`datasets` 5.1.0 is used for packaging):

```sh
python code/build.py --output rebuilt --workers 4
python code/validate_release.py --release rebuilt --report rebuilt/validation-report.json
python code/baselines.py --release rebuilt --output rebuilt/baselines
python code/package.py --release rebuilt --demo rebuilt-explorer
python code/prepare_distribution.py --build rebuilt --output distribution
python -m unittest discover -s code -p 'test_*.py'
python code/score.py --release . --split test --predictions predictions.jsonl --output scores.json
```

The builder is offline; it downloads no source material or model weights and
makes no paid calls. Fonts are bundled with their original OFL notices. PNG
bytes require the recorded font, Pillow, HarfBuzz, RAQM and FreeType versions;
content, IDs, field geometry and split construction are deterministic.

The published artifact stores the images once, embedded in Parquet. Full audit
form JSONL records refer to the `forms` configuration and `form_id`; they do not
promise separate PNG paths. A local rebuild creates raw PNGs for inspection,
then the packager and distribution step produce the bounded publication payload.

## Limits, sources and attribution

All forms are clean, digital, low-density fixtures with four recurring field
labels and three section headings. The six domains vary titles, section names,
option labels and values while sharing the same schema. They contain no natural
handwriting, scan degradation, customer documents, free-form paraphrases,
multilingual real records, nested section hierarchies or expert workflow review.
Numbers and dates are literal fictional fields; no financial, legal, medical or
historical truth is asserted. Familiarization with the public generator can solve
the controlled language; do not use these results to rank production reliability.

English is the documentation/interface default. Arabic source forms and questions
are preserved. English questions are authored controlled parallel renderings,
not independently reviewed human translations. Some checkbox labels remain Arabic
inside English requests because the model must match the displayed Arabic label.

Related work includes [XFUND](https://github.com/doc-analysis/XFUND),
[Form-NLU](https://arxiv.org/abs/2304.01577),
[KITAB-Bench](https://mbzuai-oryx.github.io/KITAB-Bench/),
[Cross-Lingual SynthDocs](https://arxiv.org/abs/2511.04699), and
[HITL Arabic OCR](https://humansintheloop.org/resources/datasets/arabic-documents-ocr-dataset/).
Those works establish form, OCR and document understanding tasks; their data is
not included here. No first, largest or human-gold claim is made.

Original fixtures, images, annotations and documentation: **CC BY 4.0**, credited
to Bayanat Labs and the exact dataset revision. Original code/explorer:
**Apache-2.0**. Bundled unmodified Noto font software: **SIL OFL 1.1**, with Noto
Project copyrights and exact hashes in [FONT_PROVENANCE.md](FONT_PROVENANCE.md).
The fonts' software license does not replace the CC BY license for original
rendered fixtures. See [DATA_LICENSE.txt](DATA_LICENSE.txt), [LICENSE](LICENSE)
and [CITATION.cff](CITATION.cff). Corrections: hello@bayanatlabs.com.

```bibtex
@misc{bayanatlabs2026arabicformgrounding,
  title = {Arabic Form Grounding},
  author = {{Bayanat Labs}},
  year = {2026},
  version = {0.1.0},
  url = {https://huggingface.co/datasets/Bayanat-Labs/arabic-form-grounding}
}
```
