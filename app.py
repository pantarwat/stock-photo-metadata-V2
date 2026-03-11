import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
import io
import zipfile
import base64
import re
import os
import json
import time
import tempfile
import hashlib
import traceback
from typing import Dict, List, Tuple, Any, Optional
from iptcinfo3 import IPTCInfo
from openai import OpenAI

# =========================
# 1) PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Stock Vision Pro", layout="wide")
st.title("🎯 AI Stock Vision Pro")
st.caption("สร้าง Title / Keywords / CSV / ZIP พร้อมฝัง IPTC Metadata สำหรับงาน Stock")

# =========================
# 2) CONSTANTS
# =========================
CATEGORY_DICT = {
    "1. Animals: สัตว์ แมลง สัตว์เลี้ยง": 1,
    "2. Buildings and Architecture: บ้าน อาคาร งานออกแบบภายใน วัด โรงงาน": 2,
    "3. Business: คนทำงาน สำนักงาน แนวคิดทางธุรกิจ การเงิน": 3,
    "4. Drinks: เครื่องดื่ม วัฒนธรรมการดื่ม แอลกอฮอล์": 4,
    "5. The Environment: ธรรมชาติ สถานที่ทำงานและที่อยู่อาศัย": 5,
    "6. States of Mind: อารมณ์ ความรู้สึก ความคิดภายในจิตใจ": 6,
    "7. Food: อาหาร การกิน วัตถุดิบ": 7,
    "8. Graphic Resources: พื้นหลัง พื้นผิว สัญลักษณ์ต่างๆ": 8,
    "9. Hobbies and Leisure: กิจกรรมยามว่าง การพักผ่อน งานอดิเรก": 9,
    "10. Industry: งานอุตสาหกรรม การผลิต พลังงาน": 10,
    "11. Landscape: ทิวทัศน์ เมือง วิวธรรมชาติ": 11,
    "12. Lifestyle: กิจกรรมในชีวิตประจำวันของคนในสถานที่ต่างๆ": 12,
    "13. People: ผู้คนทุกช่วงวัย เชื้อชาติ และความหลากหลาย": 13,
    "14. Plants and Flowers: พืชพรรณ ดอกไม้ การจัดสวน": 14,
    "15. Culture and Religion: ประเพณี ความเชื่อ วัฒนธรรมทั่วโลก": 15,
    "16. Science: วิทยาศาสตร์ การแพทย์ การวิจัย": 16,
    "17. Social Issues: ปัญหาสังคม การเมือง ความยากจน": 17,
    "18. Sports: กีฬา การออกกำลังกาย สันทนาการ": 18,
    "19. Technology: คอมพิวเตอร์ สมาร์ทโฟน AI และนวัตกรรม": 19,
    "20. Transport: ยานพาหนะ ระบบขนส่ง รถ รถไฟ เครื่องบิน": 20,
    "21. Travel: การท่องเที่ยว วัฒนธรรมท้องถิ่น สถานที่ท่องเที่ยว": 21
}

DEFAULT_MODEL_OPTIONS = {
    "GPT-5.4 (แนะนำสุด)": "gpt-5.4",
    "GPT-5-mini (เร็ว/ประหยัด)": "gpt-5-mini",
    "GPT-5-nano (เร็วมาก/ถูกมาก)": "gpt-5-nano",
    "GPT-4.1": "gpt-4.1",
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "กรอกชื่อโมเดลเอง (Custom)": "custom"
}

DEFAULT_BLACKLIST = "nike, apple, logo, brand, trademark, disney, marvel, coca-cola, adidas, samsung, sony, tesla, iphone, ipad, macbook"

KEYWORD_LIMIT = 49
TITLE_MIN_LEN = 70
TITLE_MAX_LEN = 200
IMAGE_ANALYSIS_MAX_SIDE = 1800
IMAGE_ANALYSIS_QUALITY = 90

# =========================
# 3) SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = {}

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

if "upload_map" not in st.session_state:
    st.session_state.upload_map = {}

# =========================
# 4) HELPERS
# =========================
def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    name, _ = os.path.splitext(filename)
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return (name or "image") + ".jpg"


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def unique_file_id(filename: str, file_bytes: bytes) -> str:
    return f"{sanitize_filename(filename)}__{file_hash(file_bytes)[:16]}"


def make_analysis_cache_key(
    image_bytes: bytes,
    model: str,
    hint: str,
    category_num: int,
    blacklist: List[str],
    title_style: str,
    keyword_style: str,
) -> str:
    raw = (
        image_bytes
        + model.encode("utf-8")
        + hint.encode("utf-8")
        + str(category_num).encode("utf-8")
        + ",".join(sorted(blacklist)).encode("utf-8")
        + title_style.encode("utf-8")
        + keyword_style.encode("utf-8")
    )
    return hashlib.sha256(raw).hexdigest()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_title(raw_title: str) -> str:
    title = normalize_spaces(raw_title)
    title = title.strip(" ,.;:-")
    return title


def singular_plural_dedupe_key(keyword: str) -> str:
    kw = keyword.lower().strip()
    if kw.endswith("ies") and len(kw) > 4:
        return kw[:-3] + "y"
    if kw.endswith("es") and len(kw) > 4:
        return kw[:-2]
    if kw.endswith("s") and len(kw) > 3:
        return kw[:-1]
    return kw


def normalize_keywords(raw_keywords: Any, blacklist_words: List[str], max_keywords: int = KEYWORD_LIMIT) -> str:
    blacklist_set = {normalize_spaces(x).lower() for x in blacklist_words if normalize_spaces(x)}
    parts: List[str] = []

    if isinstance(raw_keywords, list):
        parts = [str(x).strip() for x in raw_keywords if str(x).strip()]
    elif isinstance(raw_keywords, str):
        temp = raw_keywords.replace("\n", ",")
        parts = [x.strip() for x in temp.split(",") if x.strip()]
    else:
        parts = []

    cleaned = []
    seen_exact = set()
    seen_soft = set()

    for kw in parts:
        kw_norm = normalize_spaces(kw)
        kw_norm = kw_norm.strip(" ,.;:-")
        kw_key = kw_norm.lower()
        kw_soft = singular_plural_dedupe_key(kw_norm)

        if not kw_norm:
            continue
        if kw_key in blacklist_set:
            continue
        if kw_key in seen_exact:
            continue
        if kw_soft in seen_soft:
            continue
        if len(kw_norm) > 80:
            continue
        if any(ch in kw_norm for ch in ["#", "@", "/", "\\", "|", "{", "}", "[", "]"]):
            continue

        seen_exact.add(kw_key)
        seen_soft.add(kw_soft)
        cleaned.append(kw_norm)

        if len(cleaned) >= max_keywords:
            break

    return ", ".join(cleaned)


def count_keywords(keywords: str) -> int:
    return len([x.strip() for x in keywords.split(",") if x.strip()])


def validate_title(title: str, blacklist_words: List[str]) -> List[str]:
    errors = []
    title_clean = normalize_title(title)
    title_lower = title_clean.lower()

    if not title_clean:
        errors.append("Title ว่าง")
    if len(title_clean) < TITLE_MIN_LEN:
        errors.append(f"Title สั้นเกินไป (น้อยกว่า {TITLE_MIN_LEN} ตัวอักษร)")
    if len(title_clean) > TITLE_MAX_LEN:
        errors.append(f"Title ยาวเกินไป (มากกว่า {TITLE_MAX_LEN} ตัวอักษร)")
    if "," in title_clean:
        errors.append("Title ไม่ควรเป็น keyword dump หรือคั่นด้วย comma เยอะเกินไป")

    blocked = [w for w in blacklist_words if w and w.lower() in title_lower]
    if blocked:
        errors.append(f"Title มีคำต้องห้าม: {', '.join(sorted(set(blocked)))}")

    return errors


def validate_keywords(keywords: str, blacklist_words: List[str]) -> List[str]:
    errors = []
    parts = [k.strip() for k in keywords.split(",") if k.strip()]
    lowered = [k.lower() for k in parts]
    blacklist_set = {x.lower() for x in blacklist_words if x.strip()}

    if len(parts) != KEYWORD_LIMIT:
        errors.append(f"Keyword count = {len(parts)} ไม่เท่ากับ {KEYWORD_LIMIT}")

    duplicates = sorted({k for k in lowered if lowered.count(k) > 1})
    if duplicates:
        errors.append(f"มี keyword ซ้ำ: {', '.join(duplicates[:10])}")

    blocked = [k for k in lowered if k in blacklist_set]
    if blocked:
        errors.append(f"มี blacklist keyword: {', '.join(sorted(set(blocked))[:10])}")

    weird = [k for k in parts if any(ch in k for ch in ["#", "@", "/", "\\", "|", "{", "}", "[", "]"])]
    if weird:
        errors.append("มี keyword ที่มีอักขระแปลก")

    return errors


def infer_risk_notes(title: str, keywords: str) -> List[str]:
    risk_notes = []
    full_text = f"{title} {keywords}".lower()

    sensitive_terms = [
        "logo", "brand", "trademark", "copyright", "celebrity",
        "iphone", "ipad", "tesla", "nike", "adidas", "disney",
        "marvel", "coca-cola", "mcdonald", "youtube", "instagram",
        "facebook", "tiktok"
    ]
    found_sensitive = [t for t in sensitive_terms if t in full_text]
    if found_sensitive:
        risk_notes.append("อาจมีความเสี่ยงเรื่องแบรนด์/ทรัพย์สินทางปัญญา")

    people_terms = ["person", "people", "man", "woman", "child", "children", "portrait", "face"]
    if any(t in full_text for t in people_terms):
        risk_notes.append("ถ้ามีบุคคลชัดเจน อาจต้องมี model release")

    property_terms = ["home", "house", "building", "office", "interior", "property"]
    if any(t in full_text for t in property_terms):
        risk_notes.append("ถ้าเป็นทรัพย์สินเอกชนหรือภายในอาคาร อาจต้องมี property release")

    return risk_notes


def quality_score(title: str, keywords: str, title_errors: List[str], keyword_errors: List[str]) -> int:
    score = 100
    score -= len(title_errors) * 15
    score -= len(keyword_errors) * 12

    kw_count = count_keywords(keywords)
    if kw_count < KEYWORD_LIMIT:
        score -= (KEYWORD_LIMIT - kw_count)

    if len(title) < TITLE_MIN_LEN:
        score -= 8
    if len(title) > TITLE_MAX_LEN:
        score -= 8

    return max(0, min(100, score))


def optimize_image_for_analysis(image_bytes: bytes, max_size: int = IMAGE_ANALYSIS_MAX_SIDE, quality: int = IMAGE_ANALYSIS_QUALITY) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail((max_size, max_size))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()

    # ลอง parse ตรง ๆ ก่อน
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # หา block JSON ก้อนแรก
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    raise ValueError("Cannot parse JSON from model output")


def build_prompt(
    category_name: str,
    category_num: int,
    hint: str,
    blacklist_words: List[str],
    title_style: str,
    keyword_style: str,
) -> str:
    blacklist_text = ", ".join(sorted(set([x.strip() for x in blacklist_words if x.strip()]))) or "None"

    return f"""
You are a professional Adobe Stock contributor and metadata editor.

Analyze the uploaded image and produce Adobe Stock-ready metadata.

Context hint from user: {hint if hint.strip() else "None"}
Adobe category chosen by user: {category_name} (Category ID: {category_num})
Title style requested by user: {title_style}
Keyword style requested by user: {keyword_style}
Forbidden words / blacklist: {blacklist_text}

Return ONLY valid JSON.
Do not add markdown.
Do not add explanation.

JSON schema:
{{
  "title": "string",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "category_id": {category_num},
  "quality_notes": ["short note 1", "short note 2"],
  "risk_notes": ["short risk note 1", "short risk note 2"]
}}

Rules:
1) TITLE
- Write exactly 1 English title.
- Length should be about 100 to 200 characters.
- Natural, fluent, descriptive, commercially useful.
- Not a keyword dump.
- No brands, trademarks, logos, copyrighted names, or celebrity names.

2) KEYWORDS
- Generate exactly 49 English keywords.
- Order by importance, most important first.
- Relevant only.
- No brands, logos, trademarks, copyrighted names, or spam.
- Keep keywords short and stock-friendly.

3) RISK NOTES
- Mention risks only if relevant, such as:
  - visible people may require model release
  - private property/interior may require property release
  - potential trademark/logo concern
- If no risk is visible, return an empty array.

4) QUALITY NOTES
- Brief review notes about metadata quality, no more than 3 short points.

Return JSON only.
""".strip()


def call_openai_with_retry(client: OpenAI, model: str, input_payload: List[Dict[str, Any]], max_output_tokens: int = 900, retries: int = 3, sleep_seconds: float = 1.8) -> Any:
    last_error = None
    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=model,
                input=input_payload,
                max_output_tokens=max_output_tokens,
            )
            return response
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    raise last_error


def repair_json_output(client: OpenAI, model: str, broken_text: str) -> Dict[str, Any]:
    repair_prompt = f"""
Convert the following text into valid JSON only.

Required schema:
{{
  "title": "string",
  "keywords": ["keyword1", "keyword2"],
  "category_id": 0,
  "quality_notes": [],
  "risk_notes": []
}}

Text to repair:
{broken_text}
""".strip()

    response = call_openai_with_retry(
        client=client,
        model=model,
        input_payload=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": repair_prompt}]
            }
        ],
        max_output_tokens=700,
        retries=2,
    )
    raw_text = (getattr(response, "output_text", "") or "").strip()
    return extract_json_from_text(raw_text)


def analyze_image_with_openai(
    image_bytes: bytes,
    category_name: str,
    category_num: int,
    hint: str,
    api_key: str,
    model: str,
    blacklist_words: List[str],
    title_style: str,
    keyword_style: str,
) -> Dict[str, Any]:
    try:
        client = OpenAI(api_key=api_key)

        optimized_bytes = optimize_image_for_analysis(image_bytes)
        base64_image = base64.b64encode(optimized_bytes).decode("utf-8")

        prompt = build_prompt(
            category_name=category_name,
            category_num=category_num,
            hint=hint,
            blacklist_words=blacklist_words,
            title_style=title_style,
            keyword_style=keyword_style,
        )

        response = call_openai_with_retry(
            client=client,
            model=model,
            input_payload=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        },
                    ],
                }
            ],
            max_output_tokens=900,
            retries=3,
        )

        raw_text = (getattr(response, "output_text", "") or "").strip()
        if not raw_text:
            return {
                "title": "",
                "keywords": "",
                "category_id": category_num,
                "quality_notes": [],
                "risk_notes": [],
                "raw": "Model returned empty output.",
                "error": True,
            }

        try:
            data = extract_json_from_text(raw_text)
        except Exception:
            data = repair_json_output(client, model, raw_text)

        title = normalize_title(data.get("title", ""))
        keywords = normalize_keywords(data.get("keywords", []), blacklist_words)
        quality_notes = data.get("quality_notes", [])
        risk_notes = data.get("risk_notes", [])

        if not isinstance(quality_notes, list):
            quality_notes = []
        if not isinstance(risk_notes, list):
            risk_notes = []

        # เติม inferred risk จากฝั่งแอปอีกชั้น
        inferred_risks = infer_risk_notes(title, keywords)
        all_risks = list(dict.fromkeys([str(x).strip() for x in risk_notes + inferred_risks if str(x).strip()]))

        return {
            "title": title,
            "keywords": keywords,
            "category_id": category_num,
            "quality_notes": quality_notes[:3],
            "risk_notes": all_risks[:5],
            "raw": raw_text,
            "error": False,
        }

    except Exception as e:
        return {
            "title": "",
            "keywords": "",
            "category_id": category_num,
            "quality_notes": [],
            "risk_notes": [],
            "raw": f"{type(e).__name__}: {str(e)}",
            "error": True,
        }


def process_to_jpg_iptc(uploaded_file_bytes: bytes, title: str, keywords: str) -> bytes:
    tmp_path = None
    try:
        img = Image.open(io.BytesIO(uploaded_file_bytes))
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, format="JPEG", quality=100, subsampling=0)
            tmp_path = tmp.name

        try:
            info = IPTCInfo(tmp_path, force=True)
            info["object name"] = title.encode("utf-8")
            info["caption/abstract"] = title.encode("utf-8")
            keyword_list = [k.strip().encode("utf-8") for k in keywords.split(",") if k.strip()]
            info["keywords"] = keyword_list
            info.save()
        except Exception:
            pass

        with open(tmp_path, "rb") as f:
            final_bytes = f.read()

        return final_bytes

    except Exception:
        return uploaded_file_bytes

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            try:
                if os.path.exists(tmp_path + "~"):
                    os.unlink(tmp_path + "~")
            except Exception:
                pass


def safe_open_image(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True, None
    except UnidentifiedImageError:
        return False, "ไฟล์ภาพไม่ถูกต้องหรือเปิดไม่ได้"
    except Exception as e:
        return False, str(e)


def prepare_uploaded_payloads(uploaded_files) -> List[Dict[str, Any]]:
    payloads = []
    seen = set()

    for f in uploaded_files:
        file_bytes = f.read()
        fid = unique_file_id(f.name, file_bytes)

        if fid in seen:
            continue
        seen.add(fid)

        ok, err = safe_open_image(file_bytes)

        payloads.append({
            "id": fid,
            "original_name": f.name,
            "safe_name": sanitize_filename(f.name),
            "bytes": file_bytes,
            "size": len(file_bytes),
            "mime": getattr(f, "type", ""),
            "valid": ok,
            "validation_error": err,
        })

    return payloads


def upsert_result(file_id: str, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    st.session_state.results[file_id] = {
        "id": file_id,
        "original_name": payload["original_name"],
        "safe_name": payload["safe_name"],
        "bytes": payload["bytes"],
        "mime": payload["mime"],
        "size": payload["size"],
        "title": result.get("title", ""),
        "keywords": result.get("keywords", ""),
        "category_id": result.get("category_id", 0),
        "quality_notes": result.get("quality_notes", []),
        "risk_notes": result.get("risk_notes", []),
        "raw": result.get("raw", ""),
        "error": result.get("error", False),
    }


# =========================
# 5) SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ ล้างข้อมูล/เริ่มใหม่", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="ใส่ API Key หรือจะตั้งเป็น environment variable ชื่อ OPENAI_API_KEY ก็ได้"
    )

    selected_model_display = st.selectbox("🤖 Model", list(DEFAULT_MODEL_OPTIONS.keys()), index=0)
    if selected_model_display == "กรอกชื่อโมเดลเอง (Custom)":
        model_choice = st.text_input(
            "✏️ พิมพ์รหัสโมเดล",
            value="gpt-5.4",
            help="เช่น gpt-5.4, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4o"
        ).strip()
    else:
        model_choice = DEFAULT_MODEL_OPTIONS[selected_model_display]

    selected_cat_name = st.selectbox("📁 Adobe Category", list(CATEGORY_DICT.keys()), index=2)

    st.divider()

    user_hint = st.text_area(
        "💡 Context Hint",
        placeholder="เช่น: chess piece with stock graph, modern business concept, startup growth",
        height=90
    )

    title_style = st.selectbox(
        "📝 Title Style",
        ["Commercial", "Descriptive", "Minimal", "Editorial-safe", "Premium marketing"],
        index=0
    )

    keyword_style = st.selectbox(
        "🏷️ Keyword Style",
        ["Balanced", "Broad reach", "Niche specific", "SEO-friendly"],
        index=0
    )

    blacklist_raw = st.text_area(
        "🛡️ Blacklist Keywords",
        value=DEFAULT_BLACKLIST,
        height=100
    )
    blacklist = [x.strip().lower() for x in blacklist_raw.split(",") if x.strip()]

    st.divider()
    st.markdown("**แพ็กเกจที่ควรติดตั้ง**")
    st.code("pip install streamlit pandas pillow openai iptcinfo3")

# =========================
# 6) MAIN UI
# =========================
try:
    uploaded_images = st.file_uploader(
        "📸 อัปโหลดรูปภาพ",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    payloads: List[Dict[str, Any]] = []
    if uploaded_images:
        payloads = prepare_uploaded_payloads(uploaded_images)
        current_ids = [p["id"] for p in payloads]
        st.session_state.upload_map = {p["id"]: p for p in payloads}

        # ล้างผลลัพธ์เฉพาะไฟล์ที่หายไป
        for old_id in list(st.session_state.results.keys()):
            if old_id not in current_ids:
                del st.session_state.results[old_id]

        total_files = len(payloads)
        valid_files = len([p for p in payloads if p["valid"]])

        s1, s2, s3 = st.columns(3)
        s1.metric("ไฟล์ทั้งหมด", total_files)
        s2.metric("ไฟล์พร้อมวิเคราะห์", valid_files)
        s3.metric("ผลลัพธ์ในระบบ", len(st.session_state.results))

        col_a, col_b = st.columns([1, 1])

        with col_a:
            if st.button("🚀 เริ่มวิเคราะห์ทั้งหมด", use_container_width=True, type="primary"):
                if not api_key.strip():
                    st.error("❌ กรุณาใส่ OpenAI API Key ก่อน")
                elif not model_choice.strip():
                    st.error("❌ กรุณาเลือกหรือกรอกชื่อโมเดล")
                else:
                    progress = st.progress(0)
                    status = st.empty()

                    valid_payloads = [p for p in payloads if p["valid"]]
                    total = max(len(valid_payloads), 1)

                    for i, payload in enumerate(valid_payloads):
                        status.info(f"กำลังวิเคราะห์: {payload['original_name']}")

                        cache_key = make_analysis_cache_key(
                            image_bytes=payload["bytes"],
                            model=model_choice,
                            hint=user_hint,
                            category_num=CATEGORY_DICT[selected_cat_name],
                            blacklist=blacklist,
                            title_style=title_style,
                            keyword_style=keyword_style,
                        )

                        if cache_key in st.session_state.analysis_cache:
                            result = st.session_state.analysis_cache[cache_key]
                        else:
                            result = analyze_image_with_openai(
                                image_bytes=payload["bytes"],
                                category_name=selected_cat_name,
                                category_num=CATEGORY_DICT[selected_cat_name],
                                hint=user_hint,
                                api_key=api_key,
                                model=model_choice,
                                blacklist_words=blacklist,
                                title_style=title_style,
                                keyword_style=keyword_style,
                            )
                            st.session_state.analysis_cache[cache_key] = result

                        upsert_result(payload["id"], payload, result)
                        progress.progress((i + 1) / total)

                    status.success("✅ วิเคราะห์เสร็จแล้ว")

        with col_b:
            if st.button("♻️ ล้างผลวิเคราะห์ทั้งหมด", use_container_width=True):
                st.session_state.results = {}
                st.rerun()

    # =========================
    # 7) RESULTS
    # =========================
    if payloads:
        st.divider()
        st.subheader("ผลลัพธ์")

        final_data = []

        for payload in payloads:
            file_id = payload["id"]

            with st.container(border=True):
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.image(payload["bytes"], use_container_width=True)
                    st.caption(payload["original_name"])

                    st.caption(f"ขนาดไฟล์: {payload['size'] / 1024:.1f} KB")
                    st.caption(f"MIME: {payload['mime'] or 'unknown'}")

                    if not payload["valid"]:
                        st.error(payload["validation_error"] or "ไฟล์นี้เปิดไม่ได้")

                with c2:
                    if not payload["valid"]:
                        continue

                    existing = st.session_state.results.get(file_id)

                    top_action_1, top_action_2 = st.columns([1, 1])

                    with top_action_1:
                        if st.button(f"🔍 วิเคราะห์รูปนี้", key=f"analyze_one_{file_id}", use_container_width=True):
                            if not api_key.strip():
                                st.error("❌ กรุณาใส่ OpenAI API Key ก่อน")
                            else:
                                cache_key = make_analysis_cache_key(
                                    image_bytes=payload["bytes"],
                                    model=model_choice,
                                    hint=user_hint,
                                    category_num=CATEGORY_DICT[selected_cat_name],
                                    blacklist=blacklist,
                                    title_style=title_style,
                                    keyword_style=keyword_style,
                                )

                                if cache_key in st.session_state.analysis_cache:
                                    result = st.session_state.analysis_cache[cache_key]
                                else:
                                    result = analyze_image_with_openai(
                                        image_bytes=payload["bytes"],
                                        category_name=selected_cat_name,
                                        category_num=CATEGORY_DICT[selected_cat_name],
                                        hint=user_hint,
                                        api_key=api_key,
                                        model=model_choice,
                                        blacklist_words=blacklist,
                                        title_style=title_style,
                                        keyword_style=keyword_style,
                                    )
                                    st.session_state.analysis_cache[cache_key] = result

                                upsert_result(file_id, payload, result)
                                st.rerun()

                    with top_action_2:
                        if st.button(f"🗑️ ลบผลรูปนี้", key=f"clear_one_{file_id}", use_container_width=True):
                            if file_id in st.session_state.results:
                                del st.session_state.results[file_id]
                            st.rerun()

                    if not existing:
                        st.info("ยังไม่มีผลวิเคราะห์สำหรับไฟล์นี้")
                        continue

                    if existing["error"]:
                        st.error("มีปัญหาในการวิเคราะห์ไฟล์นี้")
                        with st.expander("ดู raw output / error"):
                            st.code(existing["raw"])

                    edited_title = st.text_input(
                        "Title",
                        value=existing["title"],
                        key=f"title_{file_id}",
                    )

                    edited_keywords = st.text_area(
                        "Keywords",
                        value=existing["keywords"],
                        key=f"keywords_{file_id}",
                        height=130,
                    )

                    cleaned_title = normalize_title(edited_title)
                    cleaned_keywords = normalize_keywords(edited_keywords, blacklist)

                    title_errors = validate_title(cleaned_title, blacklist)
                    keyword_errors = validate_keywords(cleaned_keywords, blacklist)
                    score = quality_score(cleaned_title, cleaned_keywords, title_errors, keyword_errors)
                    cleaned_count = count_keywords(cleaned_keywords)

                    st.session_state.results[file_id]["title"] = cleaned_title
                    st.session_state.results[file_id]["keywords"] = cleaned_keywords

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Keyword count", f"{cleaned_count}/{KEYWORD_LIMIT}")
                    m2.metric("Quality score", score)
                    m3.metric("Category", CATEGORY_DICT[selected_cat_name])

                    if score >= 85:
                        st.success("Metadata quality: ดีมาก")
                    elif score >= 65:
                        st.warning("Metadata quality: ใช้ได้ แต่ควรเช็กอีกนิด")
                    else:
                        st.error("Metadata quality: ควรแก้ก่อน export")

                    if title_errors:
                        for err in title_errors:
                            st.warning(f"Title: {err}")

                    if keyword_errors:
                        for err in keyword_errors:
                            st.warning(f"Keywords: {err}")

                    model_quality_notes = existing.get("quality_notes", [])
                    if model_quality_notes:
                        with st.expander("Quality notes"):
                            for note in model_quality_notes:
                                st.write(f"- {note}")

                    risk_notes = list(dict.fromkeys(existing.get("risk_notes", []) + infer_risk_notes(cleaned_title, cleaned_keywords)))
                    if risk_notes:
                        with st.expander("Risk / Release notes"):
                            for note in risk_notes:
                                st.write(f"- {note}")

                    safe_name = sanitize_filename(payload["original_name"])

                    final_data.append({
                        "File ID": file_id,
                        "Filename": safe_name,
                        "Title": cleaned_title,
                        "Keywords": cleaned_keywords,
                        "Category": CATEGORY_DICT[selected_cat_name],
                        "Releases": "",
                        "Quality Score": score,
                        "original_bytes": payload["bytes"],
                    })

        # =========================
        # 8) EXPORT
        # =========================
        if final_data:
            st.divider()
            st.subheader("Export")

            export_df = pd.DataFrame(final_data)[
                ["Filename", "Title", "Keywords", "Category", "Releases", "Quality Score"]
            ]

            e1, e2 = st.columns(2)

            with e1:
                csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📊 Download CSV",
                    data=csv_bytes,
                    file_name="adobe_stock_metadata.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with e2:
                if st.button("📦 สร้าง ZIP พร้อมฝัง IPTC", use_container_width=True):
                    with st.spinner("กำลังสร้าง ZIP..."):
                        zbuf = io.BytesIO()

                        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                            zf.writestr("adobe_stock_metadata.csv", csv_bytes)

                            for item in final_data:
                                final_jpg = process_to_jpg_iptc(
                                    uploaded_file_bytes=item["original_bytes"],
                                    title=item["Title"],
                                    keywords=item["Keywords"],
                                )
                                zf.writestr(item["Filename"], final_jpg)

                        zbuf.seek(0)

                        st.download_button(
                            "📂 คลิกเพื่อดาวน์โหลด ZIP",
                            data=zbuf.getvalue(),
                            file_name="adobe_stock_package.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )

            with st.expander("ดูตารางก่อนดาวน์โหลด"):
                st.dataframe(export_df, use_container_width=True)

except Exception:
    st.error("Application Error")
    st.code(traceback.format_exc())