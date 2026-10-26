#!/usr/bin/env python3
"""Deterministic original synthetic Arabic Form Grounding compiler.

All input text and fixtures are Bayanat-authored. PIL RAQM draws Arabic directly;
annotations are semantic field rectangles, never glyph/word OCR boxes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import shutil
import struct
import sys

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION, features

SOURCE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 1600
TRANSLATE = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
KEYS = ["الرمز", "العدد", "التاريخ", "الحالة"]
KEY_EN = dict(zip(KEYS, ["code", "count", "date", "status"]))
DOMAINS = [
    {"id": "lab", "title": "استمارة تنظيم المختبر", "sections": ["العينة", "الدفعة", "الجهاز"], "sections_en": ["sample", "batch", "device"], "options": ["فحص إضافي", "حفظ نسخة", "إشعار داخلي"]},
    {"id": "training", "title": "استمارة تنظيم التدريب", "sections": ["الدورة", "المجموعة", "القاعة"], "sections_en": ["course", "group", "room"], "options": ["مواد مطبوعة", "عرض مرئي", "تأكيد الحضور"]},
    {"id": "logistics", "title": "استمارة تنظيم المخزون", "sections": ["الطلب", "الشحنة", "المخزن"], "sections_en": ["request", "shipment", "store"], "options": ["تغليف إضافي", "نسخة ورقية", "إشعار الاستلام"]},
    {"id": "library", "title": "استمارة تنظيم المكتبة", "sections": ["الإعارة", "المجموعة", "الخزانة"], "sections_en": ["loan", "collection", "cabinet"], "options": ["غلاف واق", "نسخة داخلية", "تذكير بالموعد"]},
    {"id": "venue", "title": "استمارة تنظيم المرفق", "sections": ["الحجز", "التجهيز", "القاعة"], "sections_en": ["booking", "equipment", "room"], "options": ["لوحة إرشادية", "طاولة إضافية", "تأكيد التجهيز"]},
    {"id": "maintenance", "title": "استمارة تنظيم الصيانة", "sections": ["البلاغ", "الجهاز", "الفريق"], "sections_en": ["ticket", "device", "team"], "options": ["فحص لاحق", "قطعة احتياطية", "نسخة من التقرير"]},
]


def encoded(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def uid(*parts: object) -> str:
    return hashlib.sha256(encoded(parts)).hexdigest()[:20]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def split_for(layout: int) -> str:
    # The split unit is a complete section/field geometry family, not its seed.
    return "train" if layout < 16 else "validation" if layout < 20 else "test"


def panels(arrangement: int) -> list[list[int]]:
    x, y, gap, w, h = 48, 220, 24, 1184, 1100
    if arrangement == 0:
        hh = (h - gap*2)//3
        return [[x, y+i*(hh+gap), x+w, y+i*(hh+gap)+hh] for i in range(3)]
    if arrangement == 1:
        ww = (w-gap*2)//3
        return [[x+i*(ww+gap), y, x+i*(ww+gap)+ww, y+h] for i in range(3)]
    half = (w-gap)//2
    if arrangement in {2, 3}:
        short = 360
        if arrangement == 2:
            return [[x,y,x+w,y+short],[x,y+short+gap,x+half,y+h],[x+half+gap,y+short+gap,x+w,y+h]]
        return [[x,y+h-short,x+w,y+h],[x,y,x+half,y+h-short-gap],[x+half+gap,y,x+w,y+h-short-gap]]
    halfh = (h-gap)//2
    if arrangement == 4:
        return [[x,y,x+half,y+h],[x+half+gap,y,x+w,y+halfh],[x+half+gap,y+halfh+gap,x+w,y+h]]
    return [[x+half+gap,y,x+w,y+h],[x,y,x+half,y+halfh],[x,y+halfh+gap,x+half,y+h]]


def field_rects(panel: list[int], pattern: int) -> list[list[int]]:
    left, top, right, bottom = panel
    left += 14; right -= 14; top += 72; bottom -= 14
    gap = 12
    width, height = right-left, bottom-top
    if pattern == 0:
        rh = (height-gap*3)//4
        return [[left,top+i*(rh+gap),right,top+i*(rh+gap)+rh] for i in range(4)]
    if pattern == 1:
        cw, rh = (width-gap)//2, (height-gap)//2
        # Logical order follows RTL: right column then left within each row.
        return [[right-cw,top,right,top+rh],[left,top,left+cw,top+rh],
                [right-cw,top+rh+gap,right,bottom],[left,top+rh+gap,left+cw,bottom]]
    if pattern == 2:
        rh = (height-gap*2)//3
        cw = (width-gap)//2
        return [[left,top,right,top+rh],[right-cw,top+rh+gap,right,top+rh*2+gap],
                [left,top+rh+gap,left+cw,top+rh*2+gap],[left,top+rh*2+gap*2,right,bottom]]
    cw = (width-gap)//2
    rh = (height-gap*2)//3
    return [[right-cw,top,right,bottom],[left,top,left+cw,top+rh],
            [left,top+rh+gap,left+cw,top+rh*2+gap],[left,top+rh*2+gap*2,left+cw,bottom]]


def center(box: list[int]) -> tuple[float, float]:
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def contains(outer: list[int], inner: list[int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def resolve(elements: list[dict], spec: dict) -> list[dict]:
    fields = [x for x in elements if x["kind"] == "field"]
    kind = spec["type"]
    if kind == "singleton":
        return [x for x in elements if x["kind"] == "reference"]
    if kind == "checkbox":
        return [x for x in elements if x["kind"] == "checkbox" and x["label_ar"] == spec["label_ar"]]
    if spec.get("section_ar"):
        sections = [x for x in elements if x["kind"] == "section" and x["text_ar"] == spec["section_ar"]]
        assert len(sections) == 1
        fields = [x for x in fields if contains(sections[0]["bbox_px"], x["bbox_px"])]
    if kind in {"scoped_label", "unscoped_label"}:
        return [x for x in fields if x["label_ar"] == spec["label_ar"]]
    if kind == "extreme":
        axis = 0 if spec["direction"] in {"left", "right"} else 1
        values = [center(x["bbox_px"])[axis] for x in fields]
        if not values:
            return []
        desired = (min if spec["direction"] in {"left", "above"} else max)(values)
        return [x for x in fields if center(x["bbox_px"])[axis] == desired]
    if kind == "relative":
        anchors = [x for x in fields if x["label_ar"] == spec["anchor_ar"]]
        if len(anchors) != 1:
            return []
        anchor = anchors[0]
        ax, ay = center(anchor["bbox_px"])
        def allowed(x):
            cx, cy = center(x["bbox_px"])
            return {"left": cx < ax, "right": cx > ax, "above": cy < ay, "below": cy > ay}[spec["direction"]]
        pool = [x for x in fields if x is not anchor and allowed(x)]
        if not pool:
            return []
        distance = lambda x: (center(x["bbox_px"])[0]-ax)**2+(center(x["bbox_px"])[1]-ay)**2
        best = min(map(distance, pool))
        return [x for x in pool if distance(x) == best]
    raise ValueError(spec)


def question_text(spec: dict, domain: dict) -> tuple[str, str]:
    kind = spec["type"]
    sec = spec.get("section_ar")
    sec_en = domain["sections_en"][domain["sections"].index(sec)] if sec else None
    label = spec.get("label_ar")
    en_label = KEY_EN.get(label, label)
    if kind == "scoped_label":
        return (f"حدّد حقل «{label}» داخل قسم «{sec}» وأعد قيمته.",
                f"Locate the '{en_label}' field in the '{sec_en}' section and return its value.")
    if kind == "unscoped_label":
        return (f"حدّد حقل «{label}» في النموذج وأعد قيمته. لم يُحدّد قسم.",
                f"Locate the '{en_label}' field in the form and return its value. No section is specified.")
    if kind == "singleton":
        return ("حدّد حقل «مرجع النموذج» وأعد قيمته.", "Locate the 'form reference' field and return its value.")
    if kind == "checkbox":
        return (f"حدّد خيار «{label}» في شريط الخيارات وأعد حالته: محدد أو غير محدد.",
                f"Locate the option labeled «{label}» in the options strip and return its state: selected or unselected. Return the state as Arabic text: محدد or غير محدد.")
    direction_ar = {"left": "إلى يسار", "right": "إلى يمين", "above": "أعلى من", "below": "أسفل من"}[spec["direction"]]
    direction_en = {"left": "left of", "right": "right of", "above": "above", "below": "below"}[spec["direction"]]
    if kind == "relative":
        return (f"داخل قسم «{sec}»، حدّد أقرب حقل يقع مركزه {direction_ar} مركز حقل «{spec['anchor_ar']}» وأعد قيمته. قِس المسافة بين المركزين؛ التعادل يعني طلباً ملتبساً.",
                f"In the '{sec_en}' section, locate the closest field whose center is {direction_en} the center of the '{KEY_EN[spec['anchor_ar']]}' field and return its value. Use center-to-center Euclidean distance; a tie is ambiguous.")
    extreme_ar = {"left": "الأبعد إلى اليسار", "right": "الأبعد إلى اليمين", "above": "الأعلى", "below": "الأسفل"}[spec["direction"]]
    extreme_en = {"left": "leftmost", "right": "rightmost", "above": "highest", "below": "lowest"}[spec["direction"]]
    return (f"داخل قسم «{sec}»، حدّد الحقل ذي المركز {extreme_ar} وأعد قيمته. إذا تعادل أكثر من حقل فالطلب ملتبس.",
            f"In the '{sec_en}' section, locate the field with the {extreme_en} center and return its value. If multiple fields tie, the request is ambiguous.")


@lru_cache(maxsize=128)
def font(size: int, bold: bool = False, latin: bool = False) -> ImageFont.FreeTypeFont:
    name = "NotoSans-Regular.ttf" if latin else f"NotoSansArabic-{'Bold' if bold else 'Regular'}.ttf"
    return ImageFont.truetype(str(SOURCE/"fonts"/name), size, layout_engine=ImageFont.Layout.RAQM)


@lru_cache(maxsize=8)
def cmap_coverage(font_path: str) -> frozenset[int]:
    """Read Unicode cmap formats 4/12 directly; no optional font dependency."""
    data=Path(font_path).read_bytes()
    u16=lambda at:struct.unpack_from(">H",data,at)[0]
    u32=lambda at:struct.unpack_from(">I",data,at)[0]
    count=u16(4);cmap=None
    for i in range(count):
        at=12+i*16
        if data[at:at+4]==b"cmap":cmap=u32(at+8);break
    assert cmap is not None
    result=set()
    for i in range(u16(cmap+2)):
        at=cmap+4+i*8;platform_id,encoding=u16(at),u16(at+2)
        if platform_id!=0 and not (platform_id==3 and encoding in {1,10}):continue
        table=cmap+u32(at+4);fmt=u16(table)
        if fmt==12:
            for g in range(u32(table+12)):
                p=table+16+g*12;start,end,glyph=u32(p),u32(p+4),u32(p+8)
                result.update(range(start+(glyph==0),end+1))
        elif fmt==4:
            segments=u16(table+6)//2;end_at=table+14;start_at=end_at+segments*2+2
            delta_at=start_at+segments*2;range_at=delta_at+segments*2
            for s in range(segments):
                start,end=u16(start_at+s*2),u16(end_at+s*2)
                delta,offset=u16(delta_at+s*2),u16(range_at+s*2)
                for cp in range(start,end+1):
                    if cp==65535:continue
                    if offset==0:gid=(cp+delta)&65535
                    else:
                        raw=u16(range_at+s*2+offset+2*(cp-start))
                        gid=(raw+delta)&65535 if raw else 0
                    if gid:result.add(cp)
    return frozenset(result)


def draw_text(draw: ImageDraw.ImageDraw, text: str, rect: list[int], size: int = 28, bold: bool = False) -> dict:
    """Fit one line at >=18px; measure the exact raster ink before drawing."""
    l,t,r,b = rect
    direction = "rtl" if any("\u0600" <= ch <= "\u06ff" and not ch.isdigit() for ch in text) else "ltr"
    latin_font = not any("\u0600" <= ch <= "\u06ff" for ch in text)
    for px in range(size, 17, -1):
        f = font(px, bold, latin=latin_font)
        missing=sorted({ord(ch) for ch in text if ord(ch) not in cmap_coverage(f.path)})
        assert not missing, f"Unsupported glyphs {missing} in {f.path}: {text!r}"
        # Arabic-Indic numeric runs separated by hyphens can be reordered as
        # date components by Unicode bidi even with an LTR paragraph. Render
        # these nonjoining numeric codepoints explicitly from left to right,
        # preserving the declared literal ISO ordering without invisible marks.
        literal_numeric = direction == "ltr" and not latin_font
        positions=[]
        if literal_numeric:
            advance=0.0;glyph_boxes=[]
            for ch in text:
                positions.append((advance,ch))
                glyph_boxes.append(draw.textbbox((advance,0),ch,font=f,direction="ltr"))
                advance+=draw.textlength(ch,font=f,direction="ltr")
            box=(min(v[0] for v in glyph_boxes),min(v[1] for v in glyph_boxes),max(v[2] for v in glyph_boxes),max(v[3] for v in glyph_boxes))
        else:
            box = draw.textbbox((0,0), text, font=f, direction=direction)
        tw, th = box[2]-box[0], box[3]-box[1]
        if tw <= r-l and th <= b-t:
            x = r-tw-box[0] if direction == "rtl" else l-box[0]
            y = t+(b-t-th)//2-box[1]
            actual = (x+box[0],y+box[1],x+box[2],y+box[3])
            assert contains(rect, list(actual)), (text, rect, actual)
            if literal_numeric:
                for advance,ch in positions:
                    draw.text((x+advance,y),ch,font=f,fill="#172a3e",direction="ltr")
            else:
                draw.text((x,y),text,font=f,fill="#172a3e",direction=direction)
            return {"text": text, "ink_bbox_px": list(actual), "font_size_px": px, "font_file":Path(f.path).name,"glyph_coverage_checked":True,"direction": direction,"layout_strategy":"explicit_ltr_numeric_glyphs" if literal_numeric else "raqm_shaped_run", "available_bbox_px": rect}
    raise ValueError(f"Text does not fit at 18px: {text!r} in {rect}")


def render_job(payload: tuple[dict, str]) -> dict:
    form, root_text = payload
    root = Path(root_text)
    image = Image.new("RGB", (WIDTH,HEIGHT), "#edf1f5")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([24,24,WIDTH-24,HEIGHT-24],radius=22,fill="white",outline="#b2bfcc",width=2)
    ink = []
    ink.append(draw_text(draw, form["title_ar"], [48,38,1232,105], 38, True))
    ink.append(draw_text(draw,"نموذج خيالي للاختبار؛ جميع القيم مصطنعة",[48,107,1232,150],24))
    for el in form["elements"]:
        l,t,r,b = el["bbox_px"]
        if el["kind"] == "section":
            draw.rounded_rectangle([l,t,r,b],radius=12,fill="#f7f9fc",outline="#8c9eaf",width=2)
            draw.rectangle([l+2,t+2,r-2,t+58],fill=form["accent"])
            ink.append(draw_text(draw,el["text_ar"],[l+12,t+6,r-12,t+55],30,True))
        elif el["kind"] in {"field","reference"}:
            draw.rectangle([l,t,r,b],fill="white",outline="#b4bec9",width=2)
            if el["kind"] == "reference":
                mid = l+260
                ink.append(draw_text(draw,el["label_ar"],[r-250,t+4,r-12,b-4],23,True))
                ink.append(draw_text(draw,el["value_ar"],[l+12,t+4,r-270,b-4],22))
            else:
                label_h = min(44, max(30,(b-t)//2))
                ink.append(draw_text(draw,el["label_ar"],[l+8,t+4,r-8,t+label_h],23,True))
                ink.append(draw_text(draw,el["value_ar"],[l+8,t+label_h,r-8,b-5],25))
        elif el["kind"] == "checkbox":
            draw.rectangle([l,t,r,b],fill="#f7f9fc",outline="#b4bec9",width=2)
            check = [r-46,t+(b-t)//2-13,r-20,t+(b-t)//2+13]
            draw.rectangle(check,outline="#344d63",width=2)
            if el["value_ar"] == "محدد":
                draw.line([check[0]+4,check[1]+4,check[2]-4,check[3]-4],fill="#172a3e",width=3)
                draw.line([check[2]-4,check[1]+4,check[0]+4,check[3]-4],fill="#172a3e",width=3)
            ink.append(draw_text(draw,el["label_ar"],[l+8,t+8,r-58,b-8],22))
            el["control_bbox_px"] = check
    ink.append(draw_text(draw,"الخيارات: المربع المتقاطع يعني محدد؛ المربع الفارغ يعني غير محدد",[48,1330,1232,1374],22))
    ink.append(draw_text(draw,"مناطق الحقول تشمل العنوان والقيمة؛ لا تمثل حدود حروف الكلمات",[48,1490,1232,1545],21))
    relative = Path("images")/form["split"]/f"{form['form_id']}.png"
    path = root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path,optimize=False,compress_level=6)
    form["image_path"] = relative.as_posix()
    form["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    form["image_bytes"] = path.stat().st_size
    form["ink_regions"] = ink
    return form


def construct(layout: int, domain: dict, instance: int, variant: str) -> dict:
    split = split_for(layout)
    family = uid("afg-v0.1",layout,domain["id"],instance)
    form_id = uid(family,variant)
    headings = list(domain["sections"])
    if variant == "section_counterfactual":
        headings[0],headings[1] = headings[1],headings[0]
    numeral = (lambda s: str(s).translate(TRANSLATE)) if variant == "eastern_digits" else str
    missing_key = instance % 4
    elements = [{"element_id":uid(family,"reference"),"kind":"reference","label_ar":"مرجع النموذج","value_ar":f"DEMO-{split[0].upper()}{layout:02d}-{domain['id'].upper()}-{instance+1:02d}","bbox_px":[48,162,1232,205]}]
    # Panels are drawn before fields to avoid erasing already drawn content.
    boxes = panels(layout//4)
    for sec_index, box in enumerate(boxes):
        elements.append({"element_id":uid(family,"section",sec_index),"kind":"section","text_ar":headings[sec_index],"bbox_px":box})
    for sec_index, box in enumerate(boxes):
        rects = field_rects(box,layout%4)
        # Rotate label positions per content instance. This is not a split unit.
        positions = [(k+instance+sec_index)%4 for k in range(4)]
        for k,label in enumerate(KEYS):
            if sec_index == 0 and k == missing_key:
                continue
            # Values are context-specific and do not reveal which question will be asked.
            number = 10000+layout*1000+DOMAINS.index(domain)*100+instance*10+sec_index
            vals = [f"{domain['id'][:2].upper()}-{number}",str(10+((number*7+sec_index*13)%89)),
                    date(2030+(layout//16),1+(number%12),1+(number%27)).isoformat(),
                    ["جاهز","قيد الإعداد","مؤجل"][(instance+sec_index)%3]]
            value = vals[k] if k in {0,3} else numeral(vals[k])
            elements.append({"element_id":uid(family,"field",sec_index,k),"kind":"field","label_ar":label,"value_ar":value,"bbox_px":rects[positions[k]]})
    options_width = (1184-24)//3
    for option_index,label in enumerate(domain["options"]):
        x = 48+option_index*(options_width+12)
        elements.append({"element_id":uid(family,"option",option_index),"kind":"checkbox","label_ar":label,"value_ar":"محدد" if (instance+layout+option_index)%3 == 0 else "غير محدد","bbox_px":[x,1382,x+options_width,1470]})
    for el in elements:
        el["bbox"] = [round(el["bbox_px"][0]/WIDTH,7),round(el["bbox_px"][1]/HEIGHT,7),round(el["bbox_px"][2]/WIDTH,7),round(el["bbox_px"][3]/HEIGHT,7)]
        if el["kind"] != "section":
            el["text_ar"] = el["label_ar"]+"\n"+el["value_ar"]
    return {"form_id":form_id,"family_id":family,"base_context_id":family,"variant":variant,
            "layout_family":f"layout-{layout:02d}","section_arrangement":layout//4,"field_arrangement":layout%4,
            "domain":domain["id"],"split":split,"width":WIDTH,"height":HEIGHT,
            "title_ar":domain["title"],"accent":["#dce8f5","#dfecdf","#f3e6d4","#e9e0f4"][layout%4],
            "elements":elements,"provenance":"Bayanat-authored original synthetic fixture","instance":instance}


def specs_for(base: dict, domain: dict) -> list[dict]:
    instance = base["instance"]
    missing = KEYS[instance%4]
    direct_key = KEYS[(instance+1)%4]
    specs = [
        {"type":"scoped_label","section_ar":domain["sections"][0],"label_ar":direct_key},
        {"type":"scoped_label","section_ar":domain["sections"][1],"label_ar":direct_key},
    ]
    # Select a visible unique anchor and direction with at least one valid target.
    for anchor in KEYS[instance%4:]+KEYS[:instance%4]:
        found = None
        for direction in ["left","right","above","below"]:
            candidate = {"type":"relative","section_ar":domain["sections"][2],"anchor_ar":anchor,"direction":direction}
            if len(resolve(base["elements"],candidate)) == 1:
                found = candidate
                break
        if found:
            specs.append(found)
            break
    assert len(specs) == 3
    specs.extend([
        {"type":"extreme","section_ar":domain["sections"][2],"direction":["left","right","above","below"][instance%4]},
        {"type":"unscoped_label","label_ar":KEYS[(instance+2)%4]},
        {"type":"scoped_label","section_ar":domain["sections"][0],"label_ar":missing,"control":"missing_in_base"},
        {"type":"checkbox","label_ar":domain["options"][instance%3]},
        {"type":"singleton"},
    ])
    return specs


def make_questions(form: dict, specs: list[dict], domain: dict) -> list[dict]:
    result=[]
    for q_index,spec in enumerate(specs):
        matches=resolve(form["elements"],spec)
        status="missing" if not matches else "answerable" if len(matches)==1 else "ambiguous"
        target=matches[0] if len(matches)==1 else None
        ar,en=question_text(spec,domain)
        result.append({"question_id":uid(form["form_id"],q_index),"question_family_id":uid(form["family_id"],q_index),
            "form_id":form["form_id"],"family_id":form["family_id"],"split":form["split"],
            "layout_family":form["layout_family"],"domain":form["domain"],"variant":form["variant"],
            "task":spec["type"],"question_ar":ar,"question_en":en,
            "translation_status":"Bayanat-authored controlled parallel question; not independently human-translated",
            "gold_status":status,"gold_field_id":target["element_id"] if target else None,
            "gold_text":target["value_ar"] if target else None,"gold_bbox":target["bbox"] if target else None,
            "gold_bbox_px":target["bbox_px"] if target else None,
            "candidate_field_ids":[x["element_id"] for x in matches],"query_spec":spec})
    return result


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in records),encoding="utf8")


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--pilot",action="store_true")
    args=parser.parse_args()
    assert features.check("raqm"), "Pillow RAQM/HarfBuzz required for Arabic shaping"
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[];all_specs={};domains={x["id"]:x for x in DOMAINS}
    for layout in range(24):
        for domain in DOMAINS[:1] if args.pilot else DOMAINS:
            for instance in range(1 if args.pilot else 6):
                base=construct(layout,domain,instance,"base")
                all_specs[base["family_id"]]=specs_for(base,domain)
                jobs.append((base,str(root)))
                if instance==0 and not args.pilot:
                    jobs += [(construct(layout,domain,instance,v),str(root)) for v in ["eastern_digits","section_counterfactual"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        forms=[]
        for index,form in enumerate(pool.map(render_job,jobs,chunksize=4)):
            forms.append(form)
            if (index+1)%48==0 or index+1==len(jobs):
                print(f"Rendered {index+1}/{len(jobs)} forms",flush=True)
    questions=[]
    for form in forms:
        questions.extend(make_questions(form,all_specs[form["family_id"]],domains[form["domain"]]))
    for split in ["train","validation","test"]:
        write_jsonl(root/"data"/f"forms-{split}.jsonl",[x for x in forms if x["split"]==split])
        write_jsonl(root/"data"/f"questions-{split}.jsonl",[x for x in questions if x["split"]==split])
    pairs=[]
    by_q={}
    for q in questions:
        by_q.setdefault(q["question_family_id"],{})[q["variant"]]=q
    for qfamily,views in sorted(by_q.items()):
        for variant in ["eastern_digits","section_counterfactual"]:
            if variant not in views:continue
            a,b=views["base"],views[variant]
            same_target=a["gold_field_id"]==b["gold_field_id"] and a["gold_status"]==b["gold_status"]
            pairs.append({"pair_id":uid(qfamily,variant),"family_id":a["family_id"],"split":a["split"],"base_question_id":a["question_id"],"variant_question_id":b["question_id"],"variant":variant,"relation":"invariant" if same_target else "sensitive","question_text_equal":a["question_ar"]==b["question_ar"]})
    write_jsonl(root/"data"/"pairs.jsonl",pairs)
    counts={"forms":len(forms),"independent_base_contexts":len(set(x["family_id"] for x in forms)),
            "questions":len(questions),"independent_base_question_contexts":len(by_q),"pairs":len(pairs),
            "forms_by_split":dict(Counter(x["split"] for x in forms)),
            "questions_by_split":dict(Counter(x["split"] for x in questions)),
            "questions_by_status":dict(Counter(x["gold_status"] for x in questions)),
            "questions_by_task":dict(Counter(x["task"] for x in questions)),
            "forms_by_variant":dict(Counter(x["variant"] for x in forms)),
            "pairs_by_relation":dict(Counter(x["relation"] for x in pairs)),
            "layout_families":24,"fictional_domains":1 if args.pilot else 6,
            "minimum_font_px":min(i["font_size_px"] for x in forms for i in x["ink_regions"]),
            "image_bytes":sum(x["image_bytes"] for x in forms)}
    write_json(root/"statistics.json",counts)
    write_json(root/"build-manifest.json",{"version":"0.1.0","scope":"original synthetic forms and controlled questions",
        "renderer":"Pillow with RAQM/HarfBuzz; direct raster semantic field rectangles",
        "pillow":PILLOW_VERSION,"python":platform.python_version(),"raqm":features.version("raqm"),
        "harfbuzz":features.version("harfbuzz"),"freetype2":features.version("freetype2"),
        "dimensions":[WIDTH,HEIGHT],"pilot":args.pilot,
        "build_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "font_files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SOURCE/"fonts").glob("*.ttf"))},
        "determinism":"Fixture IDs/content are deterministic; PNG bytes require the recorded rendering/font/runtime versions.",
        "counts":counts})
    # Every layout has one real rendered example in this QA contact sheet.
    thumb_w,thumb_h=256,320
    sheet=Image.new("RGB",(thumb_w*6,thumb_h*4),"white")
    for layout in range(24):
        form=next(x for x in forms if x["layout_family"]==f"layout-{layout:02d}")
        with Image.open(root/form["image_path"]) as im:
            thumb=im.resize((thumb_w,thumb_h))
            sheet.paste(thumb,((layout%6)*thumb_w,(layout//6)*thumb_h))
    sheet.save(root/"layout-contact-sheet.jpg",quality=88)
    print(json.dumps(counts,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
