#!/usr/bin/env python3
"""Prepare publication payload without 1,152 redundant standalone PNGs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--build",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    source=Path(__file__).resolve().parent
    assert args.build.resolve()!=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    for path in args.build.rglob("*"):
        if not path.is_file():continue
        rel=path.relative_to(args.build)
        if rel.parts[0] in {"images","code"} or "__pycache__" in rel.parts or path.suffix==".pyc":continue
        target=args.output/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if rel.parent==Path("data") and rel.name.startswith("forms-") and rel.suffix==".jsonl":
            with path.open(encoding="utf8") as reader,target.open("w",encoding="utf8") as writer:
                for line in reader:
                    form=json.loads(line);form.pop("image_path",None);form["image_parquet_config"]="forms"
                    writer.write(json.dumps(form,ensure_ascii=False,separators=(",",":"))+"\n")
        else:shutil.copy2(path,target)
    code=args.output/"code";code.mkdir(exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix!=".pyc":
            target=code/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    print(json.dumps({"artifact":str(args.output.resolve()),"standalone_images":0,"image_configuration":"forms","source_build_retained":str(args.build.resolve())}))


if __name__=="__main__":main()
