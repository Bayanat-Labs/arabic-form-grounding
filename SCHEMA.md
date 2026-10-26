# Schema and coordinate contract

`forms` has one row per rendered form. `form_id` identifies that render;
`family_id` groups its base context and matched views. `layout_family` is a complete
section/field geometry combination. `domain`, `variant`, `width`, `height`,
`image_sha256` and `image` describe the original synthetic fixture. `image` embeds
PNG bytes with the Hugging Face Image feature. `elements_json` is a JSON array
of semantic layout areas; decode it explicitly.

Each element has `element_id`, `kind`, `text_ar`, `bbox_px` and `bbox`.
Kinds are `section`, `field`, `reference`, `checkbox`. A section rectangle encloses
its visible heading and fields. Field/reference `text_ar` joins the label and
value with a newline. Checkbox text joins the label and the renderer's visible
selected/unselected state; this is an oracle annotation, not OCR. `bbox_px` uses
integer `[left,top,right,bottom]` semantic boundaries; the right and bottom denote
the outer rectangle boundary. `bbox` divides x values by width and y by height.
Pillow draws the visible border at those boundary coordinates. No pixel-perfect
glyph coverage or token localization is asserted.

`questions` joins on `form_id` and contains `question_id`, `question_family_id`,
`family_id`, split/layout/domain/variant/task metadata, `question_ar`,
`question_en`, translation status, `gold_status`, nullable `gold_field_id`,
`gold_text`, `gold_bbox`, `gold_bbox_px`, `candidate_field_ids`, and
`query_spec_json`. The last field is an audit program, not a model input.
The JSONL source uses an object named `query_spec` instead of the serialized
Parquet `query_spec_json`. Gold is supplied in train, validation and test.

`candidate_field_ids` lists all matching fields: zero means missing, one
answerable, more than one ambiguous. The ambiguous state is about target fields,
even if two candidates happen to have the same value. Missing/ambiguous rows
have null target/text/box. `query_spec` describes the controlled predicate that
generated the request; the independent validator resolves the released Arabic
string without reading it.

`pairs` contains pair ID, family, split, base/variant question IDs, variant type,
invariant/sensitive relation and a question-text equality check. Relation means
same or changed target/status; a digit variant can change the literal value
string while preserving canonical numeric value. English is a column, not an
extra view or independent question.

The source `data/forms-*.jsonl` also retains `ink_regions`, renderer-side label
and value strings, checkbox-square rectangles, fictional instance metadata and
an `image_parquet_config` reference. Join `form_id` to the embedded image in the
`forms` configuration; the published artifact has no standalone PNG paths.
A raw local build has `image_path` instead. Ink rectangles are renderer diagnostics only. They are not the
dataset's word/line annotations. Model inputs must use `code/contract.py` to
exclude gold, predicate programs, task/variant metadata and candidate sets.
