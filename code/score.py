#!/usr/bin/env python3
"""Score predictions without an LLM judge or output repair."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import unicodedata

STATUSES={"answerable","missing","ambiguous"}
DIGITS=str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789")


def read_jsonl(path):
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():yield json.loads(line)


def canonical(text):
    return unicodedata.normalize("NFC",text).strip().translate(DIGITS)


def iou(a,b):
    if a is None or b is None:return 0.0
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.0


def valid(prediction):
    if not isinstance(prediction,dict):return False
    if not {"question_id","status","text","bbox"}<=set(prediction):return False
    if set(prediction)-{"question_id","status","text","bbox","field_id"}:return False
    if not isinstance(prediction["status"],str) or prediction["status"] not in STATUSES:return False
    if prediction.get("field_id") is not None and not isinstance(prediction["field_id"],str):return False
    if prediction["status"]!="answerable":
        return prediction["text"] is None and prediction["bbox"] is None and prediction.get("field_id") is None
    if not isinstance(prediction["text"],str) or not prediction["text"].strip():return False
    box=prediction["bbox"]
    return (isinstance(box,list) and len(box)==4 and
            all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and 0<=x<=1 for x in box) and
            box[0]<box[2] and box[1]<box[3])


def assess(gold,prediction):
    schema=prediction is not None and valid(prediction)
    state=schema and prediction["status"]==gold["gold_status"]
    answerable=gold["gold_status"]=="answerable"
    text=bool(schema and answerable and prediction["status"]=="answerable" and canonical(prediction["text"])==canonical(gold["gold_text"]))
    visible_text=bool(schema and answerable and prediction["text"]==gold["gold_text"])
    overlap=iou(prediction["bbox"],gold["gold_bbox"]) if schema and answerable and prediction["status"]=="answerable" else 0.0
    supplied_id=prediction.get("field_id") if schema else None
    id_ok=supplied_id is None or supplied_id==gold["gold_field_id"]
    joint=bool(state and id_ok and (text and overlap>=0.5 if answerable else True))
    return {"schema":bool(schema),"status":bool(state),"text":text,"visible_text":visible_text,
            "iou":overlap,"localization":overlap>=0.5,"strict_localization":overlap>=0.99,
            "field_id_ok_if_supplied":id_ok,"joint":joint,"answerable":answerable}


def aggregate(results):
    n=len(results);answered=[r for r in results if r["answerable"]]
    rate=lambda key,pool:sum(r[key] for r in pool)/len(pool) if pool else None
    return {"n":n,"answerable_n":len(answered),"schema_rate":rate("schema",results),
            "status_accuracy":rate("status",results),"canonical_text_exact_answerable":rate("text",answered),
            "visible_text_exact_answerable":rate("visible_text",answered),
            "iou_at_0_5_answerable":rate("localization",answered),"iou_at_0_99_answerable":rate("strict_localization",answered),
            "mean_iou_answerable":sum(r["iou"] for r in answered)/len(answered) if answered else None,
            "joint_success":rate("joint",results)}


def score(golds,predictions,pairs=()):
    by_id={}
    gold_ids={g["question_id"] for g in golds}
    for p in predictions:
        if not isinstance(p,dict) or not isinstance(p.get("question_id"),str):raise ValueError("Prediction needs a string question_id")
        qid=p["question_id"]
        if qid in by_id:raise ValueError(f"Duplicate prediction: {qid}")
        if qid not in gold_ids:raise ValueError(f"Unknown question_id: {qid}")
        by_id[qid]=p
    results={g["question_id"]:assess(g,by_id.get(g["question_id"])) for g in golds}
    report={"overall":aggregate(list(results.values())),"predictions_received":len(by_id),"missing_predictions":len(golds)-len(by_id)}
    for axis in ["split","task","domain","variant","gold_status","layout_family"]:
        groups=defaultdict(list)
        for g in golds:groups[g[axis]].append(results[g["question_id"]])
        report[axis]={name:aggregate(group) for name,group in sorted(groups.items())}
    report["abstention"]={}
    for status in ["missing","ambiguous"]:
        positive=[g for g in golds if g["gold_status"]==status]
        selected=[g for g in golds if by_id.get(g["question_id"],{}).get("status")==status and results[g["question_id"]]["schema"]]
        true=sum(g["gold_status"]==status for g in selected)
        report["abstention"][status]={"gold_n":len(positive),"predicted_n":len(selected),"precision":true/len(selected) if selected else None,"recall":true/len(positive) if positive else None}
    paired=defaultdict(list)
    for p in pairs:
        if p["base_question_id"] not in results or p["variant_question_id"] not in results:continue
        success=results[p["base_question_id"]]["joint"] and results[p["variant_question_id"]]["joint"]
        paired[p["variant"]+":"+p["relation"]].append(success)
    report["paired_both_joint_correct"]={name:{"n":len(values),"rate":sum(values)/len(values)} for name,values in sorted(paired.items())}
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--release",type=Path,required=True);ap.add_argument("--predictions",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--split",choices=["all","train","validation","test"],default="test");args=ap.parse_args()
    splits=["train","validation","test"] if args.split=="all" else [args.split]
    golds=[row for split in splits for row in read_jsonl(args.release/"data"/f"questions-{split}.jsonl")]
    report=score(golds,list(read_jsonl(args.predictions)),list(read_jsonl(args.release/"data/pairs.jsonl")))
    report["protocol"]={"version":"afg-score-v0.1","text_normalization":"NFC, trim and Eastern-to-Western digits only","bbox":"full semantic field area; normalized xyxy; IoU>=0.5","null_contract":"missing/ambiguous require null text/bbox/field_id","missing_prediction":"counts wrong","field_id":"optional; if supplied must equal gold target","no_output_repair":True}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report["overall"]))


if __name__=="__main__":main()
