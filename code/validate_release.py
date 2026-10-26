#!/usr/bin/env python3
"""Independent visible-text/geometry checks; does not import the builder.

Arabic queries are reparsed from the released strings; query_spec, instance,
gold candidates and generator state are not used for reference resolution.
This is programmatic validation, not independent human Arabic annotation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from PIL import Image


def rows(path):
    with path.open(encoding="utf8") as handle:
        yield from (json.loads(line) for line in handle if line.strip())


class ImageBytes:
    """Support build PNGs or publication Parquet, caching one compressed shard."""
    def __init__(self,root):
        self.root=root;self.index={};self.cache_path=None;self.cache={}
    def __call__(self,form):
        if form.get("image_path"):
            return (self.root/form["image_path"]).read_bytes()
        import pyarrow.parquet as pq
        if not self.index:
            for path in sorted((self.root/"parquet").glob("forms-*.parquet")):
                for item in pq.read_table(path,columns=["form_id"]).to_pylist():
                    assert item["form_id"] not in self.index
                    self.index[item["form_id"]]=path
        path=self.index[form["form_id"]]
        if path!=self.cache_path:
            self.cache={item["form_id"]:item["image"]["bytes"] for item in pq.read_table(path,columns=["form_id","image"]).to_pylist()}
            self.cache_path=path
        return self.cache[form["form_id"]]


def mid(box):
    return ((box[0]+box[2])/2,(box[1]+box[3])/2)


def within(container, target):
    return container[0] <= target[0] and container[1] <= target[1] and container[2] >= target[2] and container[3] >= target[3]


def visible_match(question_ar, elements):
    quoted = re.findall("«([^»]+)»",question_ar)
    # Read labels/values from text; discard the builder's separated properties.
    blocks=[{"id":e["element_id"],"kind":e["kind"],"box":e["bbox_px"],
             "label":e["text_ar"].split("\n")[0],
             "value":"\n".join(e["text_ar"].split("\n")[1:])} for e in elements]
    if question_ar.startswith("حدّد خيار"):
        return [e for e in blocks if e["kind"]=="checkbox" and e["label"]==quoted[0]]
    fields=[e for e in blocks if e["kind"] in {"field","reference"}]
    if question_ar.startswith("حدّد حقل"):
        if "داخل قسم" in question_ar:
            headings=[e for e in blocks if e["kind"]=="section" and e["label"]==quoted[1]]
            assert len(headings)==1
            fields=[e for e in fields if within(headings[0]["box"],e["box"])]
        return [e for e in fields if e["label"]==quoted[0]]
    assert question_ar.startswith("داخل قسم"), question_ar
    headings=[e for e in blocks if e["kind"]=="section" and e["label"]==quoted[0]]
    assert len(headings)==1
    fields=[e for e in fields if e["kind"]=="field" and within(headings[0]["box"],e["box"])]
    if "أقرب حقل" in question_ar:
        anchor=[e for e in fields if e["label"]==quoted[1]]
        assert len(anchor)==1
        ax,ay=mid(anchor[0]["box"])
        if "إلى يسار" in question_ar:
            options=[e for e in fields if mid(e["box"])[0]<ax]
        elif "إلى يمين" in question_ar:
            options=[e for e in fields if mid(e["box"])[0]>ax]
        elif "أعلى من" in question_ar:
            options=[e for e in fields if mid(e["box"])[1]<ay]
        else:
            assert "أسفل من" in question_ar
            options=[e for e in fields if mid(e["box"])[1]>ay]
        if not options:return []
        distances={e["id"]:(mid(e["box"])[0]-ax)**2+(mid(e["box"])[1]-ay)**2 for e in options}
        minimum=min(distances.values())
        return [e for e in options if distances[e["id"]]==minimum]
    if "الأبعد إلى اليسار" in question_ar:
        coordinate,index,fn="x",0,min
    elif "الأبعد إلى اليمين" in question_ar:
        coordinate,index,fn="x",0,max
    elif "الأعلى" in question_ar:
        coordinate,index,fn="y",1,min
    else:
        assert "الأسفل" in question_ar
        coordinate,index,fn="y",1,max
    desired=fn(mid(e["box"])[index] for e in fields)
    return [e for e in fields if mid(e["box"])[index]==desired]


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--release",type=Path,required=True);ap.add_argument("--report",type=Path,required=True);args=ap.parse_args()
    root=args.release.resolve();forms={};questions={};split_layouts=defaultdict(set);families=defaultdict(set)
    image_bytes=ImageBytes(root)
    image_hashes=Counter();validated_images=0;ink_count=0
    for split in ["train","validation","test"]:
        for f in rows(root/"data"/f"forms-{split}.jsonl"):
            assert f["split"]==split and f["form_id"] not in forms
            forms[f["form_id"]]=f;split_layouts[split].add(f["layout_family"]);families[f["family_id"]].add(split)
            raw=image_bytes(f)
            digest=hashlib.sha256(raw).hexdigest();assert digest==f["image_sha256"]
            image_hashes[digest]+=1
            with Image.open(BytesIO(raw)) as image:
                assert image.format=="PNG" and image.size==(f["width"],f["height"])
                image.load();assert image.getextrema() != ((255,255),(255,255),(255,255))
            validated_images+=1
            elements=f["elements"];ids=set()
            for e in elements:
                assert e["element_id"] not in ids;ids.add(e["element_id"])
                x0,y0,x1,y1=e["bbox_px"]
                assert 0<=x0<x1<=f["width"] and 0<=y0<y1<=f["height"]
                expected=[x0/f["width"],y0/f["height"],x1/f["width"],y1/f["height"]]
                assert all(abs(a-b)<0.0000001 for a,b in zip(expected,e["bbox"]))
                if e["kind"]=="field":
                    containers=[s for s in elements if s["kind"]=="section" and within(s["bbox_px"],e["bbox_px"])]
                    assert len(containers)==1
                if e["kind"]=="checkbox":
                    assert within(e["bbox_px"],e["control_bbox_px"])
            areas=[e for e in elements if e["kind"] in {"field","reference","checkbox"}]
            for n,a in enumerate(areas):
                for b in areas[n+1:]:
                    aa,bb=a["bbox_px"],b["bbox_px"]
                    assert min(aa[2],bb[2])<=max(aa[0],bb[0]) or min(aa[3],bb[3])<=max(aa[1],bb[1]), (a,b)
            for ink in f["ink_regions"]:
                assert ink["font_size_px"]>=18 and within(ink["available_bbox_px"],ink["ink_bbox_px"])
                ink_count+=1
        for q in rows(root/"data"/f"questions-{split}.jsonl"):
            assert q["split"]==split and q["question_id"] not in questions
            questions[q["question_id"]]=q
            form=forms[q["form_id"]]
            assert form["family_id"]==q["family_id"] and form["layout_family"]==q["layout_family"]
            assert not any(e["element_id"] in q["question_ar"] or e["element_id"] in q["question_en"] for e in form["elements"])
            candidates=visible_match(q["question_ar"],form["elements"])
            status="missing" if len(candidates)==0 else "answerable" if len(candidates)==1 else "ambiguous"
            assert status==q["gold_status"],q
            assert sorted(x["id"] for x in candidates)==sorted(q["candidate_field_ids"])
            if len(candidates)==1:
                target=candidates[0]
                assert target["id"]==q["gold_field_id"] and target["value"]==q["gold_text"] and target["box"]==q["gold_bbox_px"]
            else:
                assert q["gold_field_id"] is None and q["gold_text"] is None and q["gold_bbox"] is None
    assert not split_layouts["train"] & split_layouts["test"]
    assert not split_layouts["train"] & split_layouts["validation"]
    assert not split_layouts["validation"] & split_layouts["test"]
    assert all(len(splits)==1 for splits in families.values())
    assert len(image_hashes)==len(forms), "Exact image duplicate found"
    pair_counts=Counter();cf_changed=Counter()
    canonical=lambda s: s.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩","0123456789")) if isinstance(s,str) else s
    for p in rows(root/"data"/"pairs.jsonl"):
        a,b=questions[p["base_question_id"]],questions[p["variant_question_id"]]
        assert a["family_id"]==b["family_id"]==p["family_id"] and a["split"]==b["split"]==p["split"]
        assert a["question_ar"]==b["question_ar"] and a["question_en"]==b["question_en"]
        same=a["gold_status"]==b["gold_status"] and a["gold_field_id"]==b["gold_field_id"]
        assert (p["relation"]=="invariant")==same
        if p["variant"]=="eastern_digits":
            assert same and canonical(a["gold_text"])==canonical(b["gold_text"]) and a["gold_bbox"]==b["gold_bbox"]
        else:
            cf_changed["changed_target_or_status" if not same else "unchanged"]+=1
        pair_counts[p["variant"]]+=1
    report={"status":"passed","validator":"independent Arabic released-question parser + visible text/geometry; no builder import or query_spec use",
            "forms":len(forms),"questions":len(questions),"decoded_images":validated_images,"unique_image_sha256":len(image_hashes),
            "ink_regions_contained":ink_count,"layout_families_by_split":{k:sorted(v) for k,v in split_layouts.items()},
            "family_split_overlap":0,"pairs":dict(pair_counts),"counterfactual_effect":dict(cf_changed),
            "not_claimed":["human gold","word/glyph localization","natural form realism","OCR accuracy","model difficulty","independent native translation review"]}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":main()
