#!/usr/bin/env python3
"""Bounded Parquet/image packaging and static, English-default explorer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from datasets import Features, Image as HFImage, Value


def rows(path):
    with path.open(encoding="utf8") as handle:
        yield from (json.loads(line) for line in handle if line.strip())


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--release",type=Path,required=True);ap.add_argument("--demo",type=Path,required=True);args=ap.parse_args()
    root=args.release.resolve();demo=args.demo.resolve();demo.mkdir(parents=True,exist_ok=True)
    features=Features({"form_id":Value("string"),"family_id":Value("string"),"layout_family":Value("string"),"domain":Value("string"),"variant":Value("string"),"width":Value("int32"),"height":Value("int32"),"image":HFImage(),"image_sha256":Value("string"),"elements_json":Value("string")})
    parquet_report=[]
    for split in ["train","validation","test"]:
        forms=list(rows(root/"data"/f"forms-{split}.jsonl"))
        folder=root/"parquet";folder.mkdir(exist_ok=True)
        for offset in range(0,len(forms),96):
            batch=[]
            for f in forms[offset:offset+96]:
                batch.append({k:f[k] for k in ["form_id","family_id","layout_family","domain","variant","width","height","image_sha256"]}|{"image":{"bytes":(root/f["image_path"]).read_bytes(),"path":None},"elements_json":json.dumps(f["elements"],ensure_ascii=False,separators=(",",":"))})
            table=pa.Table.from_pylist(batch,schema=features.arrow_schema)
            path=folder/f"forms-{split}-{offset//96:03d}.parquet";pq.write_table(table,path,compression="zstd")
            check=pq.read_table(path)
            assert check.num_rows==len(batch) and check.schema.metadata[b"huggingface"]==table.schema.metadata[b"huggingface"]
            for r in check.to_pylist():assert hashlib.sha256(r["image"]["bytes"]).hexdigest()==r["image_sha256"]
            parquet_report.append({"path":path.relative_to(root).as_posix(),"rows":len(batch),"image_bytes_sha256_checked":True})
        qs=list(rows(root/"data"/f"questions-{split}.jsonl"))
        # Keep public audit programs as JSON strings, not accidental model inputs.
        packed=[{k:v for k,v in q.items() if k!="query_spec"}|{"query_spec_json":json.dumps(q["query_spec"],ensure_ascii=False,separators=(",",":"))} for q in qs]
        path=folder/f"questions-{split}.parquet";pq.write_table(pa.Table.from_pylist(packed),path,compression="zstd")
        assert pq.read_metadata(path).num_rows==len(qs)
        parquet_report.append({"path":path.relative_to(root).as_posix(),"rows":len(qs)})
        pairs=[p for p in rows(root/"data/pairs.jsonl") if p["split"]==split]
        path=folder/f"pairs-{split}.parquet";pq.write_table(pa.Table.from_pylist(pairs),path,compression="zstd")
        parquet_report.append({"path":path.relative_to(root).as_posix(),"rows":len(pairs)})
    write_json(root/"parquet-validation.json",{"status":"passed","shards":parquet_report,"maximum_form_rows_in_memory":96,"image_storage":"PNG compressed bytes embedded, no external image download required"})

    forms=[f for split in ["train","validation","test"] for f in rows(root/"data"/f"forms-{split}.jsonl")]
    questions=[q for split in ["train","validation","test"] for q in rows(root/"data"/f"questions-{split}.jsonl")]
    chosen=set()
    domains=["lab","training","logistics","library","venue","maintenance"]
    for index in range(24):
        layout=f"layout-{index:02d}";domain=domains[index%6]
        base=next(f for f in forms if f["layout_family"]==layout and f["domain"]==domain and f["variant"]=="base" and f["instance"]==0)
        chosen.update(f["form_id"] for f in forms if f["family_id"]==base["family_id"])
    (demo/"images").mkdir(exist_ok=True)
    for f in forms:
        if f["form_id"] not in chosen:continue
        shutil.copy2(root/f["image_path"],demo/"images"/f"{f['form_id']}.png")
        with Image.open(root/f["image_path"]) as image:
            image.resize((256,320)).save(demo/"images"/f"{f['form_id']}-thumb.webp",quality=82)
    examples=[]
    keep={"relative","unscoped_label"}
    for q in questions:
        if q["form_id"] not in chosen:continue
        if q["task"] in keep or q["task"]=="scoped_label" and (q["query_spec"].get("control")=="missing_in_base" or q["query_spec"].get("label_ar")=="العدد"):
            examples.append({k:q[k] for k in ["question_id","question_family_id","form_id","family_id","split","layout_family","domain","variant","task","question_ar","question_en","gold_status","gold_text","gold_bbox","gold_field_id","candidate_field_ids"]}|{"image":f"images/{q['form_id']}.png","thumbnail":f"images/{q['form_id']}-thumb.webp"})
    write_json(demo/"examples.json",examples)
    write_json(demo/"statistics.json",json.loads((root/"statistics.json").read_text()))
    shutil.copy2(root/"layout-contact-sheet.jpg",demo/"layout-contact-sheet.jpg")
    (root/".gitattributes").write_text("*.parquet filter=lfs diff=lfs merge=lfs -text\n*.png filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n")
    (demo/".gitattributes").write_text("*.png filter=lfs diff=lfs merge=lfs -text\n*.jpg filter=lfs diff=lfs merge=lfs -text\n*.webp filter=lfs diff=lfs merge=lfs -text\n")
    print(json.dumps({"parquet_shards":len(parquet_report),"demo_forms":len(chosen),"demo_question_rows":len(examples)}))


if __name__=="__main__":main()
