#!/usr/bin/env python3
"""Transparent given-layout rules baselines; no OCR/model/hidden query program.

lexical: exact field label matching, ignores section scopes and spatial requests.
section_geometry: additionally parses a named section and uses visible panel
containment; still has no nearest/extreme relation solver. Both get exact
rendered semantic text/rectangles, including checkbox states. They are not image
model baselines and do not measure OCR.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from contract import oracle_layout_input
from score import read_jsonl, score


def run(record,method):
    text=record["question"];quotes=re.findall("«([^»]+)»",text)
    blocks=[]
    for e in record["elements"]:
        lines=e["text_ar"].split("\n")
        blocks.append({**e,"label":lines[0],"value":"\n".join(lines[1:])})
    candidates=[]
    if text.startswith("حدّد خيار"):
        candidates=[e for e in blocks if e["kind"]=="checkbox" and e["label"]==quotes[0]]
    elif text.startswith("حدّد حقل"):
        candidates=[e for e in blocks if e["kind"] in {"field","reference"} and e["label"]==quotes[0]]
        if method=="section_geometry" and "داخل قسم" in text:
            panels=[e for e in blocks if e["kind"]=="section" and e["label"]==quotes[1]]
            if len(panels)==1:
                b=panels[0]["bbox"]
                candidates=[e for e in candidates if b[0]<=e["bbox"][0] and b[1]<=e["bbox"][1] and b[2]>=e["bbox"][2] and b[3]>=e["bbox"][3]]
    # Unsupported spatial queries explicitly abstain. This is a baseline failure,
    # not a ground-truth claim that these visible queries have no answer.
    status="missing" if not candidates else "answerable" if len(candidates)==1 else "ambiguous"
    target=candidates[0] if len(candidates)==1 else None
    return {"question_id":record["question_id"],"status":status,
            "text":target["value"] if target else None,"bbox":target["bbox"] if target else None,
            "field_id":target["element_id"] if target else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--release",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    forms={f["form_id"]:f for split in ["train","validation","test"] for f in read_jsonl(args.release/"data"/f"forms-{split}.jsonl")}
    golds=[q for split in ["train","validation","test"] for q in read_jsonl(args.release/"data"/f"questions-{split}.jsonl")]
    pairs=list(read_jsonl(args.release/"data/pairs.jsonl"));args.output.mkdir(parents=True,exist_ok=True)
    summaries={}
    for method in ["lexical","section_geometry"]:
        predictions=[run(oracle_layout_input(q,forms[q["form_id"]]),method) for q in golds]
        (args.output/f"{method}-predictions.jsonl").write_text("".join(json.dumps(p,ensure_ascii=False)+"\n" for p in predictions))
        report=score(golds,predictions,pairs)
        report["baseline_contract"]={"method":method,"track":"oracle semantic layout and text","no_ocr_or_model":True,"no_query_spec_or_gold_in_input":True,"unsupported_spatial_queries":"returns missing and is scored as a failure when answerable/ambiguous"}
        (args.output/f"{method}-report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        summaries[method]={"overall":report["overall"],"test":report["split"]["test"],"paired":report["paired_both_joint_correct"]}
    (args.output/"summary.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=="__main__":main()
