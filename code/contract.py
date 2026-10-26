"""Public input and output contracts. Gold and generator programs stay out."""
from __future__ import annotations
import json

OUTPUT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["status", "text", "bbox"],
    "properties": {
        "status": {"type": "string", "enum": ["answerable", "missing", "ambiguous"]},
        "text": {"type": ["string", "null"]},
        "bbox": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}]},
        "field_id": {"type": ["string", "null"]},
    },
}

INSTRUCTIONS = """Return one JSON object: status (answerable, missing, ambiguous),
text (the requested visible value or Arabic checkbox state), bbox (normalized
[left, top, right, bottom], each 0..1), and optional field_id. When exactly one
semantic field satisfies the request, return answerable and the full rectangular
field area including its label and value. A checkbox area includes its label and
square. Return missing for no matching field, ambiguous for multiple matching
fields, and null text/bbox/field_id for either abstention. Center comparisons use
field rectangle centers; closest means Euclidean distance and exact ties are
ambiguous. Respect visible section headings. Do not guess missing values.
The code/date/count text is literal fixture content. Keep opaque identifiers
unchanged. English questions are controlled parallel renderings of the Arabic
question; forms remain Arabic. This is fictional material, not a real record."""


def image_input(question: dict, form: dict, language: str = "ar") -> dict:
    return {"instructions": INSTRUCTIONS, "question_id": question["question_id"],
            "question": question["question_ar" if language == "ar" else "question_en"],
            "form_id":form["form_id"],"image_config":"forms",
            "image":form.get("image"),
            "image_path": form.get("image_path"), "width": form["width"], "height": form["height"]}


def oracle_layout_input(question: dict, form: dict, language: str = "ar") -> dict:
    """The oracle-layout track supplies exact semantic text/areas, not OCR."""
    elements=form["elements"] if "elements" in form else json.loads(form["elements_json"])
    return {**image_input(question, form, language),
            "elements": [{"element_id": element["element_id"], "kind": element["kind"],
                          "text_ar": element["text_ar"], "bbox": element["bbox"]}
                         for element in elements]}
