# Arabic Form Grounding — source v0.1.0

An offline compiler, independent validator and scorer for **original synthetic Arabic forms**. Locate a field from its section, label or spatial relationship, return its value and complete semantic field rectangle, and abstain when no single answer is supported. The full release has 864 base contexts, 1,152 rendered views, 9,216 question rows and 2,304 matched pairs. Repeated views are not independent real forms. No customer records, natural scans, human annotation or learned-model accuracy are claimed.

[Dataset v0.1.0](https://huggingface.co/datasets/Bayanat-Labs/arabic-form-grounding/tree/v0.1.0) · [Explorer](https://huggingface.co/spaces/Bayanat-Labs/arabic-form-grounding) · [Company release page](https://www.bayanatlabs.com/open/arabic-form-grounding) · [Full dataset card](DATASET_CARD.md)

This checkout contains code, unit tests, schema/rights documents and permitted unmodified fonts. Released JSONL/Parquet and images live on Hugging Face and are absent here. The full dataset card describes that download, not files already present in this source checkout.

## Quickstart: bounded offline pilot

Use Python 3.14. The tested renderer uses Pillow 12.3.0 with RAQM/HarfBuzz; requirements also pin full-release packaging libraries. No model/API is called and the builder performs no network acquisition.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python -c "from PIL import features; assert features.check('raqm')"
python -m unittest discover -s code -p 'test_*.py' -v
python code/build.py --pilot --output work/pilot --workers 2
python code/validate_release.py --release work/pilot --report work/pilot/validation-report.json
python code/baselines.py --release work/pilot --output work/pilot/baselines
```

The pilot renders **24 forms and 192 questions with no matched view pairs**, covering all 24 layouts with one domain/instance each. It checks the renderer and visible-query contract; it is not full-release evaluation. CI runs these unit tests, the shaping check, bounded pilot and independent validator. Fonts are bundled with original OFL notices. Content/IDs/geometry are deterministic; PNG bytes also depend on the recorded rendering versions.

## Complete rebuild and packaging

```sh
python code/build.py --output work/full --workers 4
python code/validate_release.py --release work/full --report work/full/validation-report.json
python code/baselines.py --release work/full --output work/full/baselines
python code/package.py --release work/full --demo work/full-explorer
python code/prepare_distribution.py --build work/full --output work/distribution
```

The builder writes PNGs plus audit JSONL. The packager embeds PNG bytes in Parquet and makes an explorer sample payload. The distribution step removes redundant standalone PNGs and adjusts form references, without changing question meaning. All outputs remain under ignored `work/`. These code steps create data assets; release cards/licenses/citations accompany the public dataset separately.

## Model boundary and evaluation

[SCHEMA.md](SCHEMA.md) defines the task. `code/contract.py` builds safe image-only or given-layout inputs. Gold fields, candidate IDs, programs, variants and family metadata must remain hidden. Given-layout inputs expose exact semantic text/rectangles and checkbox states; that track is not OCR. All gold is public and may be contaminated.

Write one JSONL prediction per `question_id`, then run:

```sh
python code/score.py --release work/full --split test --predictions predictions.jsonl --output work/scores.json
```

Predictions contain `status` (`answerable`, `missing`, `ambiguous`), `text` and normalized `[left,top,right,bottom]` `bbox`; abstention requires null text/bbox. The box covers the complete semantic field, including label/value, not tight glyphs. The scorer reports exact canonical text, IoU, abstention and paired metrics. Missing/invalid predictions are wrong; duplicate/unknown IDs fail the submission. There is no LLM judge or repair.

Full-test given-layout baselines score 37.5% for label-only matching and 75.0% for section-aware matching. Both receive oracle text/rectangles; the second deliberately lacks spatial-relation handling. The gap is a designed limitation, not evidence of learned-model difficulty. No OCR/vision-model/LLM score is claimed. Grammar and components recur; layout-family holdouts are a narrow diagnostic, not broad generalization evidence.

## Licenses and credits

Original code/explorer: **Apache-2.0** ([LICENSE](LICENSE)). Original rendered fixtures, annotations and documentation: **CC BY 4.0**, credited to Bayanat Labs and the exact revision ([DATA_LICENSE.txt](DATA_LICENSE.txt)). Bundled unmodified Noto font software: **SIL OFL 1.1**, with Noto Project copyrights, full notice and exact hashes in [FONT_PROVENANCE.md](FONT_PROVENANCE.md). Font software licensing does not replace the fixture-image license.

These clean, controlled, low-density fixtures do not represent handwriting, degraded scans, customer records or real financial/legal/medical facts. English is the documentation/interface default; Arabic form strings are preserved. English questions are controlled authored counterparts, not human translations. Related work is credited in the full card; no third-party benchmark examples are included and no first/largest claim is made. Cite [CITATION.cff](CITATION.cff).
