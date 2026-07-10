import base64
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo
from iptcinfo3 import IPTCInfo
from openai import OpenAI


# =========================================================
# 1) PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Stock Vision Pro",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 AI Stock Vision Pro")
st.caption(
    "สร้าง Title จาก 10 Keywords แรก พร้อม Export CSV / ZIP "
    "และรักษาชนิดไฟล์ JPG, JPEG, PNG ตามต้นฉบับ"
)


# =========================================================
# 2) CONFIG
# =========================================================
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
    "21. Travel: การท่องเที่ยว วัฒนธรรมท้องถิ่น สถานที่ท่องเที่ยว": 21,
}

# Official docs currently list GPT-5.6 as the latest model family.
MODEL_OPTIONS = {
    "GPT-5.6 — Latest": "gpt-5.6",
    "Custom model ID": "custom",
}

DEFAULT_BLACKLIST = (
    "nike, apple, adidas, disney, marvel, coca-cola, samsung, sony, "
    "tesla, iphone, ipad, macbook, logo, trademark, brand, celebrity"
)

KEYWORD_LIMIT = 49
TOP_KEYWORD_COUNT = 10
MIN_TOP_KEYWORD_COVERAGE = 5
TITLE_MIN_LENGTH = 70
TITLE_MAX_LENGTH = 200
ANALYSIS_MAX_SIDE = 1800
ANALYSIS_JPEG_QUALITY = 90
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# IMPORTANT:
# The server cleanup button deletes ONLY this app-owned directory.
APP_CACHE_DIR = Path(
    os.getenv(
        "AI_STOCK_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "ai_stock_vision_cache"),
    )
)
UPLOAD_CACHE_DIR = APP_CACHE_DIR / "uploads"
EXPORT_CACHE_DIR = APP_CACHE_DIR / "exports"
ANALYSIS_CACHE_DIR = APP_CACHE_DIR / "analysis"

for folder in (UPLOAD_CACHE_DIR, EXPORT_CACHE_DIR, ANALYSIS_CACHE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3) SESSION STATE
# =========================================================
DEFAULT_SESSION_STATE = {
    "results": {},
    "analysis_cache": {},
    "title_cache": {},
    "uploader_version": 0,
    "generated_zip": None,
    "generated_zip_name": "",
    "flash_message": "",
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 4) GENERAL HELPERS
# =========================================================
def normalize_spaces(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_title(value: Any) -> str:
    return normalize_spaces(value).strip(" ,;:-")


def parse_blacklist(raw_text: str) -> List[str]:
    return [
        normalize_spaces(item).lower()
        for item in raw_text.split(",")
        if normalize_spaces(item)
    ]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def original_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return ".jpg" if ext == ".jpe" else ext


def sanitize_filename(filename: str) -> str:
    base = os.path.basename(filename)
    path = Path(base)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        ext = ".jpg"

    stem = re.sub(r"[^\w\-. ]+", "_", path.stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip(" ._-") or "image"
    return f"{stem}{ext}"


def unique_file_id(filename: str, data: bytes) -> str:
    return f"{sanitize_filename(filename)}__{sha256_bytes(data)[:16]}"


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def directory_stats(directory: Path) -> Tuple[int, int]:
    if not directory.exists():
        return 0, 0

    file_count = 0
    total_bytes = 0

    for path in directory.rglob("*"):
        if path.is_file():
            file_count += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    return file_count, total_bytes


def cache_image_on_server(file_id: str, filename: str, data: bytes) -> Path:
    """
    Store one app-owned copy on server so the dedicated server-cleanup
    button has a real directory to manage.
    """
    file_dir = UPLOAD_CACHE_DIR / file_id
    file_dir.mkdir(parents=True, exist_ok=True)
    target = file_dir / sanitize_filename(filename)

    if not target.exists():
        target.write_bytes(data)

    return target


def clear_page_uploads() -> None:
    """
    Button 1:
    Clear only current browser/session page state.
    Do NOT delete server cache directories.
    """
    for file_id in list(st.session_state.results.keys()):
        st.session_state.pop(f"title_{file_id}", None)
        st.session_state.pop(f"keywords_{file_id}", None)

    st.session_state.results = {}
    st.session_state.generated_zip = None
    st.session_state.generated_zip_name = ""
    st.session_state.uploader_version += 1
    st.session_state.flash_message = "ล้างภาพออกจากหน้าอัปโหลดเรียบร้อยแล้ว"


def clear_server_cache() -> Tuple[int, int]:
    """
    Button 2:
    Delete all app-owned files under APP_CACHE_DIR and clear in-memory caches.
    It intentionally does NOT reset the file uploader or current page images.
    """
    file_count, total_bytes = directory_stats(APP_CACHE_DIR)

    if APP_CACHE_DIR.exists():
        shutil.rmtree(APP_CACHE_DIR)

    for folder in (UPLOAD_CACHE_DIR, EXPORT_CACHE_DIR, ANALYSIS_CACHE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    st.session_state.analysis_cache = {}
    st.session_state.title_cache = {}
    st.session_state.generated_zip = None
    st.session_state.generated_zip_name = ""

    return file_count, total_bytes


# =========================================================
# 5) IMAGE HELPERS
# =========================================================
def validate_image(data: bytes, filename: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    ext = original_extension(filename)

    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"ไม่รองรับไฟล์ {ext or 'ไม่ทราบชนิด'}", {}

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}[ext]

            if image_format != expected:
                return (
                    False,
                    f"นามสกุลเป็น {ext} แต่ข้อมูลภายในเป็น {image_format or 'unknown'}",
                    {},
                )

            has_transparency = (
                image.mode in ("RGBA", "LA")
                or (image.mode == "P" and "transparency" in image.info)
            )

            return True, None, {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image_format,
                "has_transparency": has_transparency,
            }

    except UnidentifiedImageError:
        return False, "ไฟล์นี้ไม่ใช่ภาพที่รองรับ", {}
    except Exception as error:
        return False, f"เปิดไฟล์ไม่ได้: {error}", {}


def prepare_uploads(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    seen = set()

    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        file_id = unique_file_id(uploaded.name, data)

        if file_id in seen:
            continue

        seen.add(file_id)
        valid, error, info = validate_image(data, uploaded.name)

        cached_path = None
        if valid:
            cached_path = cache_image_on_server(file_id, uploaded.name, data)

        payloads.append({
            "id": file_id,
            "original_name": uploaded.name,
            "safe_name": sanitize_filename(uploaded.name),
            "extension": original_extension(uploaded.name),
            "bytes": data,
            "size": len(data),
            "mime": getattr(uploaded, "type", ""),
            "valid": valid,
            "validation_error": error,
            "image_info": info,
            "server_cache_path": str(cached_path) if cached_path else "",
        })

    return payloads


def optimize_for_analysis(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)

        if image.mode in ("RGBA", "LA"):
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            image = white.convert("RGB")
        elif image.mode == "P" and "transparency" in image.info:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            image = white.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image.thumbnail(
            (ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE),
            Image.Resampling.LANCZOS,
        )

        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=ANALYSIS_JPEG_QUALITY,
            optimize=True,
            subsampling=0,
        )
        return buffer.getvalue()


# =========================================================
# 6) KEYWORDS / TITLE
# =========================================================
def split_keywords(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [normalize_spaces(item) for item in raw if normalize_spaces(item)]

    if isinstance(raw, str):
        return [
            normalize_spaces(item)
            for item in raw.replace("\n", ",").split(",")
            if normalize_spaces(item)
        ]

    return []


def normalize_keywords(raw: Any, blacklist: List[str], limit: int = KEYWORD_LIMIT) -> str:
    blocked = {item.lower() for item in blacklist}
    cleaned: List[str] = []
    seen = set()

    for keyword in split_keywords(raw):
        keyword = keyword.strip(" ,;:.")
        key = keyword.lower()

        if not keyword or key in seen or key in blocked:
            continue
        if len(keyword) > 80:
            continue
        if any(char in keyword for char in "#@\\|{}[]<>"):
            continue

        seen.add(key)
        cleaned.append(keyword)

        if len(cleaned) >= limit:
            break

    return ", ".join(cleaned)


def keyword_list(keywords: str) -> List[str]:
    return [normalize_spaces(item) for item in keywords.split(",") if normalize_spaces(item)]


def keyword_count(keywords: str) -> int:
    return len(keyword_list(keywords))


def normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def top_keywords_in_title(title: str, keywords: str) -> List[str]:
    title_words = set(normalize_match_text(title).split())
    matched: List[str] = []

    for keyword in keyword_list(keywords)[:TOP_KEYWORD_COUNT]:
        parts = [part for part in normalize_match_text(keyword).split() if len(part) >= 3]
        if not parts:
            continue

        required = 1 if len(parts) == 1 else min(2, len(parts))
        if sum(part in title_words for part in parts) >= required:
            matched.append(keyword)

    return matched


# =========================================================
# 7) OPENAI
# =========================================================
def extract_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("ไม่พบ JSON ในผลลัพธ์")

    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("JSON ไม่ใช่ object")

    return value


def call_openai(
    client: OpenAI,
    model: str,
    input_payload: List[Dict[str, Any]],
    max_output_tokens: int,
    retries: int = 3,
) -> Any:
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            return client.responses.create(
                model=model,
                input=input_payload,
                max_output_tokens=max_output_tokens,
            )
        except Exception as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))

    raise last_error or RuntimeError("OpenAI request failed")


def metadata_prompt(
    category_name: str,
    category_num: int,
    hint: str,
    blacklist: List[str],
) -> str:
    return f"""
You are a professional Adobe Stock metadata editor.

Analyze the image accurately.

Selected category: {category_name}
Category ID: {category_num}
User context: {hint.strip() if hint.strip() else "None"}
Forbidden words: {", ".join(blacklist) if blacklist else "None"}

PROCESS
1. Generate exactly 49 English keywords ordered from most important to least important.
2. The first 10 keywords must represent the primary subject, action, setting,
   visual attributes, and strongest commercial concepts.
3. After finalizing the keywords, write one natural English title using the
   meaning and search intent of the first 10 keywords.
4. The title must read naturally and must not be a pasted keyword list.

TITLE RULES
- One English title.
- Prefer 100–200 characters; never exceed 200.
- Put the main subject and action early.
- Natural, clear, descriptive, commercially useful.
- No brand names, logos, celebrity names, copyrighted names, or unsupported details.
- Do not begin with "Image of", "Photo of", or "Picture of".

KEYWORD RULES
- Exactly 49 English keywords.
- Most important first.
- Relevant, concise, and stock-search friendly.
- Avoid irrelevant filler and unnecessary duplicates.
- No brands or trademarks.

Return valid JSON only:
{{
  "title": "natural English title",
  "keywords": ["keyword 1", "keyword 2"],
  "quality_notes": [],
  "risk_notes": []
}}
""".strip()


def title_only_prompt(
    keywords: str,
    current_title: str,
    hint: str,
    blacklist: List[str],
) -> str:
    top_ten = ", ".join(keyword_list(keywords)[:TOP_KEYWORD_COUNT])

    return f"""
You are an Adobe Stock title editor.

Top 10 keywords in priority order:
{top_ten}

Context:
{hint.strip() if hint.strip() else "None"}

Current title:
{current_title.strip() if current_title.strip() else "None"}

Forbidden words:
{", ".join(blacklist) if blacklist else "None"}

Write one NEW English stock title using the meaning and strongest search intent
of the 10 keywords. Use the earliest keywords most strongly.

The title must:
- sound natural and easy to understand
- not be a pasted keyword list
- put the main subject and action early
- preferably be 100–200 characters
- never exceed 200 characters
- avoid unsupported details, brands, trademarks, and copyrighted names
- differ meaningfully from the current title

Return valid JSON only:
{{"title": "new natural English stock title"}}
""".strip()


def analyze_image(
    data: bytes,
    api_key: str,
    model: str,
    category_name: str,
    category_num: int,
    hint: str,
    blacklist: List[str],
) -> Dict[str, Any]:
    try:
        client = OpenAI(api_key=api_key)
        optimized = optimize_for_analysis(data)
        image_b64 = base64.b64encode(optimized).decode("utf-8")

        response = call_openai(
            client=client,
            model=model,
            input_payload=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": metadata_prompt(
                        category_name, category_num, hint, blacklist
                    )},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",
                    },
                ],
            }],
            max_output_tokens=1500,
        )

        raw = (response.output_text or "").strip()
        parsed = extract_json(raw)

        return {
            "title": normalize_title(parsed.get("title", "")),
            "keywords": normalize_keywords(parsed.get("keywords", []), blacklist),
            "quality_notes": parsed.get("quality_notes", [])[:3]
                if isinstance(parsed.get("quality_notes", []), list) else [],
            "risk_notes": parsed.get("risk_notes", [])[:5]
                if isinstance(parsed.get("risk_notes", []), list) else [],
            "raw": raw,
            "error": False,
            "error_message": "",
        }

    except Exception as error:
        return {
            "title": "",
            "keywords": "",
            "quality_notes": [],
            "risk_notes": [],
            "raw": f"{type(error).__name__}: {error}",
            "error": True,
            "error_message": f"{type(error).__name__}: {error}",
        }


def regenerate_title(
    keywords: str,
    current_title: str,
    api_key: str,
    model: str,
    hint: str,
    blacklist: List[str],
) -> Dict[str, Any]:
    if keyword_count(keywords) < TOP_KEYWORD_COUNT:
        return {
            "title": current_title,
            "error": True,
            "error_message": "ต้องมีอย่างน้อย 10 Keywords ก่อน",
        }

    try:
        client = OpenAI(api_key=api_key)

        response = call_openai(
            client=client,
            model=model,
            input_payload=[{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": title_only_prompt(
                        keywords, current_title, hint, blacklist
                    ),
                }],
            }],
            max_output_tokens=400,
        )

        raw = (response.output_text or "").strip()
        title = normalize_title(extract_json(raw).get("title", ""))

        if not title:
            raise ValueError("โมเดลไม่ได้สร้าง Title")

        if len(title) > TITLE_MAX_LENGTH:
            title = title[:TITLE_MAX_LENGTH].rsplit(" ", 1)[0].rstrip(" ,;:-.")

        return {"title": title, "error": False, "error_message": ""}

    except Exception as error:
        return {
            "title": current_title,
            "error": True,
            "error_message": f"{type(error).__name__}: {error}",
        }


# =========================================================
# 8) EXPORT
# =========================================================
def export_jpeg(data: bytes, title: str, keywords: str, ext: str) -> bytes:
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpeg" if ext == ".jpeg" else ".jpg",
            delete=False,
        ) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        info = IPTCInfo(temp_path, force=True)
        info["object name"] = title.encode("utf-8")
        info["caption/abstract"] = title.encode("utf-8")
        info["keywords"] = [
            item.encode("utf-8") for item in keyword_list(keywords)
        ]
        info.save()

        return Path(temp_path).read_bytes()

    except Exception:
        return data

    finally:
        if temp_path:
            for path in (Path(temp_path), Path(temp_path + "~")):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


def export_png(data: bytes, title: str, keywords: str) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            original_size = source.size
            original_has_alpha = source.mode in ("RGBA", "LA") or (
                source.mode == "P" and "transparency" in source.info
            )

            metadata = PngInfo()
            for key, value in source.info.items():
                if isinstance(value, str) and key.lower() not in {
                    "title", "description", "caption", "keywords"
                }:
                    try:
                        metadata.add_text(key, value)
                    except Exception:
                        pass

            metadata.add_text("Title", title)
            metadata.add_text("Description", title)
            metadata.add_text("Keywords", keywords)

            save_options: Dict[str, Any] = {
                "format": "PNG",
                "pnginfo": metadata,
                "compress_level": 6,
            }

            if source.info.get("icc_profile"):
                save_options["icc_profile"] = source.info["icc_profile"]
            if source.info.get("dpi"):
                save_options["dpi"] = source.info["dpi"]
            if source.mode in ("P", "L", "RGB") and "transparency" in source.info:
                save_options["transparency"] = source.info["transparency"]

            output = io.BytesIO()
            source.save(output, **save_options)
            exported = output.getvalue()

        with Image.open(io.BytesIO(exported)) as check:
            exported_has_alpha = check.mode in ("RGBA", "LA") or (
                check.mode == "P" and "transparency" in check.info
            )

            if check.size != original_size:
                raise ValueError("PNG dimensions changed")
            if original_has_alpha and not exported_has_alpha:
                raise ValueError("PNG transparency was lost")

        return exported

    except Exception:
        return data


def export_image(data: bytes, ext: str, title: str, keywords: str) -> bytes:
    if ext in {".jpg", ".jpeg"}:
        return export_jpeg(data, title, keywords, ext)
    if ext == ".png":
        return export_png(data, title, keywords)
    return data


# =========================================================
# 9) SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )

    selected_model = st.selectbox(
        "🤖 Model",
        options=list(MODEL_OPTIONS.keys()),
    )

    if MODEL_OPTIONS[selected_model] == "custom":
        model_choice = st.text_input("Model ID", value="gpt-5.6").strip()
    else:
        model_choice = MODEL_OPTIONS[selected_model]

    category_name = st.selectbox(
        "📁 Adobe Category",
        options=list(CATEGORY_DICT.keys()),
        index=2,
    )
    category_num = CATEGORY_DICT[category_name]

    hint = st.text_area(
        "💡 Context Hint",
        placeholder="เช่น young woman stretching before tennis workout",
    )

    blacklist_raw = st.text_area(
        "🛡️ Blacklist Keywords",
        value=DEFAULT_BLACKLIST,
    )
    blacklist = parse_blacklist(blacklist_raw)

    st.divider()
    server_files, server_bytes = directory_stats(APP_CACHE_DIR)
    st.caption(
        f"Server cache: {server_files:,} files • {human_size(server_bytes)}"
    )
    st.caption(f"Cache directory: {APP_CACHE_DIR}")

    confirm_server_cleanup = st.checkbox(
        "ยืนยันว่าต้องการล้างแคชทั้งหมดบนเซิร์ฟเวอร์",
        value=False,
    )

    if st.button(
        "🧹 ล้างแคชภาพทั้งหมดบนเซิร์ฟเวอร์",
        use_container_width=True,
        disabled=not confirm_server_cleanup,
    ):
        deleted_files, deleted_bytes = clear_server_cache()
        st.success(
            f"ล้างแล้ว {deleted_files:,} ไฟล์ คืนพื้นที่ {human_size(deleted_bytes)}"
        )


# =========================================================
# 10) MAIN UI
# =========================================================
try:
    if st.session_state.flash_message:
        st.success(st.session_state.flash_message)
        st.session_state.flash_message = ""

    uploader_key = f"image_uploader_{st.session_state.uploader_version}"

    uploaded_files = st.file_uploader(
        "📸 อัปโหลดรูปภาพ JPG, JPEG หรือ PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    payloads: List[Dict[str, Any]] = (
        prepare_uploads(uploaded_files) if uploaded_files else []
    )

    page_col1, page_col2 = st.columns([1, 1])

    with page_col1:
        analyze_all = st.button(
            "🚀 วิเคราะห์ภาพทั้งหมด",
            type="primary",
            use_container_width=True,
            disabled=not payloads,
        )

    with page_col2:
        clear_page = st.button(
            "🗑️ ล้างภาพออกจากหน้าอัปโหลด",
            use_container_width=True,
            disabled=not payloads,
            help="ล้างเฉพาะภาพและผลในหน้าปัจจุบัน ไม่ลบแคชบนเซิร์ฟเวอร์",
        )

    if clear_page:
        clear_page_uploads()
        st.rerun()

    if analyze_all:
        if not api_key:
            st.error("กรุณาใส่ OpenAI API Key")
        else:
            valid_payloads = [item for item in payloads if item["valid"]]
            progress = st.progress(0)
            status = st.empty()

            for index, payload in enumerate(valid_payloads, start=1):
                status.info(
                    f"กำลังวิเคราะห์ {index}/{len(valid_payloads)}: "
                    f"{payload['original_name']}"
                )

                result = analyze_image(
                    data=payload["bytes"],
                    api_key=api_key,
                    model=model_choice,
                    category_name=category_name,
                    category_num=category_num,
                    hint=hint,
                    blacklist=blacklist,
                )

                st.session_state.results[payload["id"]] = {
                    **payload,
                    **result,
                    "category": category_num,
                }
                st.session_state[f"title_{payload['id']}"] = result["title"]
                st.session_state[f"keywords_{payload['id']}"] = result["keywords"]

                progress.progress(index / max(len(valid_payloads), 1))

            status.success("วิเคราะห์เรียบร้อย")
            st.rerun()

    final_items: List[Dict[str, Any]] = []

    for payload in payloads:
        with st.container(border=True):
            image_col, detail_col = st.columns([1, 2], gap="large")

            with image_col:
                st.image(payload["bytes"], use_container_width=True)
                st.caption(payload["original_name"])

                if payload["valid"]:
                    info = payload["image_info"]
                    st.caption(
                        f"{info['width']:,} × {info['height']:,} px • "
                        f"{info['mode']} • {human_size(payload['size'])}"
                    )
                    if info["has_transparency"]:
                        st.success("PNG โปร่งใส")
                else:
                    st.error(payload["validation_error"])

            with detail_col:
                if not payload["valid"]:
                    continue

                file_id = payload["id"]
                result = st.session_state.results.get(file_id)

                if not result:
                    st.info("ยังไม่ได้วิเคราะห์ภาพนี้")
                    continue

                if result.get("error"):
                    st.error(result.get("error_message", "เกิดข้อผิดพลาด"))
                    with st.expander("Error detail"):
                        st.code(result.get("raw", ""))

                title_key = f"title_{file_id}"
                keywords_key = f"keywords_{file_id}"

                if title_key not in st.session_state:
                    st.session_state[title_key] = result.get("title", "")
                if keywords_key not in st.session_state:
                    st.session_state[keywords_key] = result.get("keywords", "")

                edited_title = st.text_area("Title", key=title_key, height=90)
                edited_keywords = st.text_area(
                    "Keywords",
                    key=keywords_key,
                    height=150,
                )

                cleaned_title = normalize_title(edited_title)
                cleaned_keywords = normalize_keywords(edited_keywords, blacklist)

                st.session_state.results[file_id]["title"] = cleaned_title
                st.session_state.results[file_id]["keywords"] = cleaned_keywords

                button_col1, button_col2 = st.columns(2)

                with button_col1:
                    if st.button(
                        "✨ สร้าง Title ใหม่เฉพาะภาพนี้",
                        key=f"new_title_{file_id}",
                        use_container_width=True,
                    ):
                        if not api_key:
                            st.error("กรุณาใส่ OpenAI API Key")
                        else:
                            title_result = regenerate_title(
                                keywords=cleaned_keywords,
                                current_title=cleaned_title,
                                api_key=api_key,
                                model=model_choice,
                                hint=hint,
                                blacklist=blacklist,
                            )

                            if title_result["error"]:
                                st.error(title_result["error_message"])
                            else:
                                st.session_state.results[file_id]["title"] = (
                                    title_result["title"]
                                )
                                st.session_state[title_key] = title_result["title"]
                                st.rerun()

                with button_col2:
                    if st.button(
                        "🗑️ ลบผลวิเคราะห์ภาพนี้",
                        key=f"remove_result_{file_id}",
                        use_container_width=True,
                    ):
                        st.session_state.results.pop(file_id, None)
                        st.session_state.pop(title_key, None)
                        st.session_state.pop(keywords_key, None)
                        st.rerun()

                used_top = top_keywords_in_title(cleaned_title, cleaned_keywords)

                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Keywords", f"{keyword_count(cleaned_keywords)}/49")
                metric2.metric("Top 10 ใน Title", f"{len(used_top)}/10")
                metric3.metric("Title length", len(cleaned_title))

                with st.expander("ตรวจสอบ 10 Keywords แรก"):
                    top_ten = keyword_list(cleaned_keywords)[:10]
                    st.write("10 Keywords แรก:")
                    st.write(", ".join(top_ten) or "-")
                    st.write("คำที่พบใน Title:")
                    st.write(", ".join(used_top) or "-")

                final_items.append({
                    "Filename": payload["safe_name"],
                    "Title": cleaned_title,
                    "Keywords": cleaned_keywords,
                    "Category": category_num,
                    "Releases": "",
                    "bytes": payload["bytes"],
                    "extension": payload["extension"],
                })

    if final_items:
        st.divider()
        st.subheader("📦 Export")

        export_df = pd.DataFrame(final_items)[
            ["Filename", "Title", "Keywords", "Category", "Releases"]
        ]
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "📊 Download CSV",
                data=csv_bytes,
                file_name="adobe_stock_metadata.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            if st.button(
                "📦 สร้าง ZIP พร้อม Metadata",
                use_container_width=True,
            ):
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(
                    zip_buffer,
                    "w",
                    zipfile.ZIP_DEFLATED,
                ) as archive:
                    archive.writestr("adobe_stock_metadata.csv", csv_bytes)

                    for item in final_items:
                        exported = export_image(
                            data=item["bytes"],
                            ext=item["extension"],
                            title=item["Title"],
                            keywords=item["Keywords"],
                        )
                        archive.writestr(item["Filename"], exported)

                zip_bytes = zip_buffer.getvalue()
                export_name = (
                    f"adobe_stock_package_{int(time.time())}.zip"
                )
                export_path = EXPORT_CACHE_DIR / export_name
                export_path.write_bytes(zip_bytes)

                st.session_state.generated_zip = zip_bytes
                st.session_state.generated_zip_name = export_name

        if st.session_state.generated_zip:
            st.download_button(
                "📂 Download ZIP",
                data=st.session_state.generated_zip,
                file_name=st.session_state.generated_zip_name,
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

        with st.expander("ดูตาราง CSV"):
            st.dataframe(export_df, use_container_width=True, hide_index=True)

except Exception:
    st.error("Application Error")
    with st.expander("รายละเอียด Error", expanded=True):
        st.code(traceback.format_exc())