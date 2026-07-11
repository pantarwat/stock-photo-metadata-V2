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
    "สร้างและตรวจคุณภาพ Adobe Stock Metadata • "
    "49 Keywords • 10 Keywords แรกต้องอยู่ใน Title ครบ 10/10"
)


# =========================================================
# 2) CONSTANTS
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

DEFAULT_BLACKLIST = (
    "nike, apple, adidas, disney, marvel, coca-cola, samsung, sony, "
    "tesla, iphone, ipad, macbook, logo, trademark, celebrity"
)

KEYWORD_LIMIT = 49
TOP_KEYWORD_COUNT = 10
TITLE_MIN_LENGTH = 70
TITLE_MAX_LENGTH = 200
MAX_TITLE_REPAIR_ATTEMPTS = 3
ANALYSIS_MAX_SIDE = 1800
ANALYSIS_JPEG_QUALITY = 90
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# แอปลบได้เฉพาะโฟลเดอร์นี้เท่านั้น
APP_CACHE_DIR = Path(
    os.getenv(
        "AI_STOCK_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "ai_stock_vision_cache"),
    )
)
UPLOAD_CACHE_DIR = APP_CACHE_DIR / "uploads"
EXPORT_CACHE_DIR = APP_CACHE_DIR / "exports"

for directory in (UPLOAD_CACHE_DIR, EXPORT_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3) SESSION STATE
# =========================================================
DEFAULT_STATE = {
    "results": {},
    "analysis_cache": {},
    "title_cache": {},
    "quality_cache": {},
    "uploader_version": 0,
    "generated_zip": None,
    "generated_zip_name": "",
    "flash_message": "",
    "model_refresh_version": 0,
}

for state_key, default_value in DEFAULT_STATE.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# =========================================================
# 4) SECURITY / GENERAL HELPERS
# =========================================================
def redact_secrets(value: Any) -> str:
    text = str(value or "")
    for pattern in (
        r"sk-proj-[A-Za-z0-9_-]+",
        r"sk-[A-Za-z0-9_-]+",
    ):
        text = re.sub(pattern, "[REDACTED_OPENAI_API_KEY]", text)
    return text


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


def stable_hash(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def original_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return ".jpg" if extension == ".jpe" else extension


def sanitize_filename(filename: str) -> str:
    path = Path(os.path.basename(filename))
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        extension = ".jpg"

    stem = re.sub(r"[^\w\-. ]+", "_", path.stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip(" ._-") or "image"
    return f"{stem}{extension}"


def unique_file_id(filename: str, data: bytes) -> str:
    return f"{sanitize_filename(filename)}__{sha256_bytes(data)[:16]}"


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def directory_stats(directory: Path) -> Tuple[int, int]:
    if not directory.exists():
        return 0, 0

    file_count = 0
    total_bytes = 0

    for path in directory.rglob("*"):
        if not path.is_file():
            continue

        file_count += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass

    return file_count, total_bytes


def cache_upload_on_server(file_id: str, filename: str, data: bytes) -> Path:
    target_directory = UPLOAD_CACHE_DIR / file_id
    target_directory.mkdir(parents=True, exist_ok=True)
    target = target_directory / sanitize_filename(filename)

    if not target.exists():
        target.write_bytes(data)

    return target


def clear_page_uploads() -> None:
    """
    ปุ่มที่ 1: ล้างเฉพาะภาพและผลบนหน้าปัจจุบัน
    ไม่ลบแคชหรือไฟล์ใน APP_CACHE_DIR
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
    ปุ่มที่ 2: ลบแคชและไฟล์ของแอปบนเซิร์ฟเวอร์
    ไม่รีเซ็ตภาพที่กำลังแสดงใน file uploader
    """
    file_count, total_bytes = directory_stats(APP_CACHE_DIR)

    if APP_CACHE_DIR.exists():
        shutil.rmtree(APP_CACHE_DIR)

    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    st.session_state.analysis_cache = {}
    st.session_state.title_cache = {}
    st.session_state.quality_cache = {}
    st.session_state.generated_zip = None
    st.session_state.generated_zip_name = ""

    return file_count, total_bytes


# =========================================================
# 5) MODEL DISCOVERY
# =========================================================
def get_available_models(api_key: str) -> Tuple[List[str], str]:
    """
    Models API บอกโมเดลที่ Key เข้าถึงได้ แต่ไม่ได้รับรองว่า
    ทุกโมเดลในรายการรองรับ input_image
    """
    if not api_key.strip():
        return [], ""

    cache_key = stable_hash({
        "key_tail": api_key[-8:],
        "refresh": st.session_state.model_refresh_version,
    })
    data_key = f"models_{cache_key}"
    error_key = f"models_error_{cache_key}"

    if data_key in st.session_state:
        return (
            st.session_state[data_key],
            st.session_state.get(error_key, ""),
        )

    try:
        client = OpenAI(api_key=api_key.strip())
        response = client.models.list()

        model_ids = sorted({
            model.id
            for model in response.data
            if getattr(model, "id", "")
        })

        excluded = (
            "embedding",
            "moderation",
            "whisper",
            "tts",
            "audio",
            "realtime",
            "transcribe",
            "dall-e",
            "sora",
        )

        candidates = [
            model_id
            for model_id in model_ids
            if model_id.startswith("gpt-")
            and not any(token in model_id.lower() for token in excluded)
        ]

        preferred_exact = [
            "gpt-5",
            "gpt-5-mini",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ]

        def sort_key(model_id: str) -> Tuple[int, str]:
            if model_id in preferred_exact:
                return preferred_exact.index(model_id), model_id

            for index, prefix in enumerate(preferred_exact):
                if model_id.startswith(prefix):
                    return index + 20, model_id

            return 100, model_id

        candidates.sort(key=sort_key)

        st.session_state[data_key] = candidates
        st.session_state[error_key] = ""
        return candidates, ""

    except Exception as error:
        message = redact_secrets(f"{type(error).__name__}: {error}")
        st.session_state[data_key] = []
        st.session_state[error_key] = message
        return [], message


def friendly_api_error(error: Exception, model: str) -> str:
    text = redact_secrets(str(error))
    lower = text.lower()

    if "model_not_found" in lower or "does not exist" in lower:
        return (
            f"ไม่พบโมเดล '{model}' หรือ API Key ไม่มีสิทธิ์ใช้งาน "
            "กรุณารีเฟรชรายการโมเดลแล้วเลือกใหม่"
        )

    if "invalid_api_key" in lower or "incorrect api key" in lower:
        return "API Key ไม่ถูกต้อง ถูกยกเลิก หรือหมดอายุ"

    if "insufficient_quota" in lower or "billing" in lower:
        return "บัญชี API ไม่มีโควตา หรือยังไม่ได้ตั้งค่า Billing"

    if "image" in lower and (
        "unsupported" in lower or "not support" in lower
    ):
        return (
            f"โมเดล '{model}' ไม่รองรับ Image Input "
            "กรุณาเลือกโมเดลที่รองรับการวิเคราะห์ภาพ"
        )

    return f"{type(error).__name__}: {text}"


# =========================================================
# 6) IMAGE HELPERS
# =========================================================
def validate_image(
    data: bytes,
    filename: str,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    extension = original_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        return False, f"ไม่รองรับไฟล์ {extension or 'ไม่ทราบชนิด'}", {}

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        with Image.open(io.BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            expected_format = {
                ".jpg": "JPEG",
                ".jpeg": "JPEG",
                ".png": "PNG",
            }[extension]

            if image_format != expected_format:
                return (
                    False,
                    f"นามสกุลเป็น {extension} แต่ข้อมูลภายในเป็น "
                    f"{image_format or 'unknown'}",
                    {},
                )

            has_transparency = (
                image.mode in ("RGBA", "LA")
                or (
                    image.mode == "P"
                    and "transparency" in image.info
                )
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
        return False, redact_secrets(f"เปิดไฟล์ไม่ได้: {error}"), {}


def prepare_uploads(uploaded_files: List[Any]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    seen_ids = set()

    for uploaded_file in uploaded_files:
        data = uploaded_file.getvalue()
        file_id = unique_file_id(uploaded_file.name, data)

        if file_id in seen_ids:
            continue

        seen_ids.add(file_id)
        valid, validation_error, image_info = validate_image(
            data,
            uploaded_file.name,
        )

        cache_path = ""
        if valid:
            cache_path = str(
                cache_upload_on_server(
                    file_id,
                    uploaded_file.name,
                    data,
                )
            )

        payloads.append({
            "id": file_id,
            "original_name": uploaded_file.name,
            "safe_name": sanitize_filename(uploaded_file.name),
            "extension": original_extension(uploaded_file.name),
            "bytes": data,
            "size": len(data),
            "mime": getattr(uploaded_file, "type", ""),
            "valid": valid,
            "validation_error": validation_error,
            "image_info": image_info,
            "server_cache_path": cache_path,
        })

    return payloads


def optimize_for_analysis(data: bytes) -> bytes:
    """
    ย่อเฉพาะสำเนาที่ส่ง AI
    ไฟล์ต้นฉบับสำหรับ Export ไม่ถูก resize
    """
    with Image.open(io.BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source)

        if image.mode in ("RGBA", "LA"):
            rgba = image.convert("RGBA")
            background = Image.new(
                "RGBA",
                rgba.size,
                (255, 255, 255, 255),
            )
            background.alpha_composite(rgba)
            image = background.convert("RGB")

        elif image.mode == "P" and "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new(
                "RGBA",
                rgba.size,
                (255, 255, 255, 255),
            )
            background.alpha_composite(rgba)
            image = background.convert("RGB")

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
# 7) KEYWORD / TITLE VALIDATION
# =========================================================
def split_keywords(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [
            normalize_spaces(item)
            for item in raw
            if normalize_spaces(item)
        ]

    if isinstance(raw, str):
        return [
            normalize_spaces(item)
            for item in raw.replace("\n", ",").split(",")
            if normalize_spaces(item)
        ]

    return []


def contains_blacklist_term(
    text: str,
    blacklist: List[str],
) -> bool:
    normalized_text = f" {re.sub(r'[^a-z0-9 ]+', ' ', text.lower())} "

    for blocked in blacklist:
        normalized_blocked = re.sub(
            r"[^a-z0-9 ]+",
            " ",
            blocked.lower(),
        ).strip()

        if normalized_blocked and (
            f" {normalized_blocked} " in normalized_text
        ):
            return True

    return False


def normalize_keywords(
    raw: Any,
    blacklist: List[str],
    limit: int = KEYWORD_LIMIT,
) -> str:
    cleaned: List[str] = []
    seen = set()

    for keyword in split_keywords(raw):
        keyword = keyword.strip(" ,;:.")
        key = keyword.lower()

        if not keyword or key in seen:
            continue
        if contains_blacklist_term(keyword, blacklist):
            continue
        if len(keyword) > 80:
            continue
        if any(character in keyword for character in "#@\\|{}[]<>"):
            continue

        seen.add(key)
        cleaned.append(keyword)

        if len(cleaned) >= limit:
            break

    return ", ".join(cleaned)


def keyword_list(keywords: str) -> List[str]:
    return [
        normalize_spaces(item)
        for item in keywords.split(",")
        if normalize_spaces(item)
    ]


def keyword_count(keywords: str) -> int:
    return len(keyword_list(keywords))


def canonical_search_text(text: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower(),
    ).strip()


def keyword_exactly_in_title(keyword: str, title: str) -> bool:
    """
    ตรวจ keyword แบบ exact phrase โดยไม่สนตัวพิมพ์ใหญ่เล็ก
    และยอมให้ punctuation คั่นคำได้
    """
    keyword_normalized = canonical_search_text(keyword)
    title_normalized = canonical_search_text(title)

    if not keyword_normalized or not title_normalized:
        return False

    return (
        f" {keyword_normalized} "
        in f" {title_normalized} "
    )


def top_keyword_coverage(
    title: str,
    keywords: str,
) -> Dict[str, Any]:
    top_ten = keyword_list(keywords)[:TOP_KEYWORD_COUNT]
    found = [
        keyword
        for keyword in top_ten
        if keyword_exactly_in_title(keyword, title)
    ]
    missing = [
        keyword
        for keyword in top_ten
        if keyword not in found
    ]

    return {
        "top_ten": top_ten,
        "found": found,
        "missing": missing,
        "count": len(found),
        "complete": (
            len(top_ten) == TOP_KEYWORD_COUNT
            and len(found) == TOP_KEYWORD_COUNT
        ),
    }


def duplicate_keywords(keywords: str) -> List[str]:
    values = [
        item.lower()
        for item in keyword_list(keywords)
    ]

    return sorted({
        item
        for item in values
        if values.count(item) > 1
    })


def deterministic_quality_check(
    title: str,
    keywords: str,
    blacklist: List[str],
    ai_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    คะแนนเต็ม 100:
    - Keywords ครบ 49: 20
    - Top 10 อยู่ใน Title ครบ: 35
    - Title length: 10
    - ไม่มี Keyword ซ้ำ: 10
    - ไม่มี Blacklist: 10
    - AI relevance/naturalness: 15
    """
    cleaned_title = normalize_title(title)
    cleaned_keywords = normalize_keywords(
        keywords,
        blacklist=[],
        limit=999,
    )

    count = keyword_count(cleaned_keywords)
    duplicates = duplicate_keywords(cleaned_keywords)
    coverage = top_keyword_coverage(
        cleaned_title,
        cleaned_keywords,
    )

    title_blacklisted = contains_blacklist_term(
        cleaned_title,
        blacklist,
    )

    blocked_keywords = [
        keyword
        for keyword in keyword_list(cleaned_keywords)
        if contains_blacklist_term(keyword, blacklist)
    ]

    title_length_ok = (
        TITLE_MIN_LENGTH
        <= len(cleaned_title)
        <= TITLE_MAX_LENGTH
    )

    score_breakdown = {
        "keyword_count": 20 if count == KEYWORD_LIMIT else max(
            0,
            20 - abs(KEYWORD_LIMIT - count) * 2,
        ),
        "top_10_coverage": round(
            35 * (coverage["count"] / TOP_KEYWORD_COUNT)
        ) if coverage["top_ten"] else 0,
        "title_length": 10 if title_length_ok else 0,
        "no_duplicates": 10 if not duplicates else 0,
        "no_blacklist": 10 if (
            not title_blacklisted
            and not blocked_keywords
        ) else 0,
        "ai_quality": 0,
    }

    ai_relevance = None
    ai_naturalness = None
    ai_accuracy = None
    ai_notes: List[str] = []

    if ai_review:
        ai_relevance = int(ai_review.get("relevance_score", 0) or 0)
        ai_naturalness = int(ai_review.get("naturalness_score", 0) or 0)
        ai_accuracy = int(ai_review.get("keyword_accuracy_score", 0) or 0)
        ai_notes = [
            normalize_spaces(note)
            for note in ai_review.get("notes", [])
            if normalize_spaces(note)
        ][:5]

        average_ai = (
            ai_relevance
            + ai_naturalness
            + ai_accuracy
        ) / 3

        score_breakdown["ai_quality"] = round(
            15 * (average_ai / 10)
        )

    total_score = min(
        100,
        sum(score_breakdown.values()),
    )

    hard_requirements = {
        "keywords_49": count == KEYWORD_LIMIT,
        "top_10_complete": coverage["complete"],
        "title_within_200": (
            0 < len(cleaned_title) <= TITLE_MAX_LENGTH
        ),
        "no_duplicates": not duplicates,
        "no_blacklist": (
            not title_blacklisted
            and not blocked_keywords
        ),
        "ai_relevance_pass": (
            ai_relevance is None
            or ai_relevance >= 7
        ),
        "ai_naturalness_pass": (
            ai_naturalness is None
            or ai_naturalness >= 7
        ),
        "ai_keyword_accuracy_pass": (
            ai_accuracy is None
            or ai_accuracy >= 7
        ),
    }

    all_hard_pass = all(hard_requirements.values())

    if all_hard_pass and total_score >= 90:
        status = "ผ่าน — พร้อม Export"
        level = "success"
    elif (
        coverage["complete"]
        and count == KEYWORD_LIMIT
        and total_score >= 75
    ):
        status = "ควรตรวจสอบเพิ่มเติม"
        level = "warning"
    else:
        status = "ไม่ผ่าน — ควรแก้ก่อน Export"
        level = "error"

    issues: List[str] = []

    if count != KEYWORD_LIMIT:
        issues.append(
            f"Keywords มี {count}/{KEYWORD_LIMIT} คำ"
        )

    if not coverage["complete"]:
        issues.append(
            "10 Keywords แรกยังอยู่ใน Title ไม่ครบ: "
            + ", ".join(coverage["missing"])
        )

    if not title_length_ok:
        issues.append(
            f"Title ควรยาว {TITLE_MIN_LENGTH}–"
            f"{TITLE_MAX_LENGTH} ตัวอักษร "
            f"(ปัจจุบัน {len(cleaned_title)})"
        )

    if duplicates:
        issues.append(
            "มี Keyword ซ้ำ: "
            + ", ".join(duplicates[:10])
        )

    if title_blacklisted or blocked_keywords:
        issues.append(
            "พบคำ Blacklist ใน Title หรือ Keywords"
        )

    if ai_relevance is not None and ai_relevance < 7:
        issues.append("AI ประเมินว่า Metadata ยังไม่ตรงภาพพอ")

    if ai_naturalness is not None and ai_naturalness < 7:
        issues.append("AI ประเมินว่า Title ยังไม่เป็นธรรมชาติพอ")

    if ai_accuracy is not None and ai_accuracy < 7:
        issues.append("AI ประเมินว่า Keywords บางคำอาจไม่แม่นยำ")

    return {
        "score": total_score,
        "status": status,
        "level": level,
        "score_breakdown": score_breakdown,
        "hard_requirements": hard_requirements,
        "issues": issues,
        "keyword_count": count,
        "duplicates": duplicates,
        "blocked_keywords": blocked_keywords,
        "title_length": len(cleaned_title),
        "title_length_ok": title_length_ok,
        "coverage": coverage,
        "ai_relevance": ai_relevance,
        "ai_naturalness": ai_naturalness,
        "ai_keyword_accuracy": ai_accuracy,
        "ai_notes": ai_notes,
    }


# =========================================================
# 8) OPENAI HELPERS
# =========================================================
def extract_json(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip()
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("ไม่พบ JSON ในผลลัพธ์ของโมเดล")

    parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("JSON output ไม่ใช่ object")

    return parsed


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
                time.sleep(1.5 * (2**attempt))

    raise last_error or RuntimeError("OpenAI request failed")


def image_data_url(data: bytes) -> str:
    optimized = optimize_for_analysis(data)
    encoded = base64.b64encode(optimized).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# =========================================================
# 9) PROMPTS
# =========================================================
def metadata_prompt(
    category_name: str,
    category_num: int,
    hint: str,
    blacklist: List[str],
) -> str:
    return f"""
You are a senior Adobe Stock metadata editor.

Analyze the image accurately and create high-quality searchable metadata.

SELECTED CATEGORY
- {category_name}
- Category ID: {category_num}

OPTIONAL USER CONTEXT
{hint.strip() if hint.strip() else "None"}

FORBIDDEN TERMS
{", ".join(blacklist) if blacklist else "None"}

MANDATORY WORKFLOW

STEP 1 — KEYWORDS
Generate exactly 49 unique English keywords ordered from most important
to least important.

The first 10 keywords have a special requirement:
- They must be the strongest search terms for the visible image.
- Keep them concise enough to fit naturally inside one title.
- Prefer single words or short phrases.
- Do not make the first 10 synonyms of the same concept.
- Together they should cover the subject, action, setting,
  visual attributes, and commercial concept.
- Every one of these first 10 keywords must later appear verbatim
  in the title, ignoring capitalization and punctuation.

STEP 2 — TITLE
After the 49 keywords are finalized, write one English stock title.

The title must:
- Contain every first-10 keyword verbatim.
- Remain natural, grammatical, clear, and easy to understand.
- Never look like a raw keyword list.
- Put the main subject and action early.
- Prefer 100–200 characters.
- Never exceed 200 characters.
- Avoid excessive commas.
- Not begin with “Image of”, “Photo of”, or “Picture of”.
- Not add details unsupported by the visible image.
- Not contain brands, trademarks, logos, celebrities,
  copyrighted characters, or forbidden terms.

Before returning:
- Count all keywords and confirm there are exactly 49.
- Confirm the first 10 all appear verbatim in the title.
- Rewrite internally until both conditions are true.

Return valid JSON only:

{{
  "title": "natural stock title containing all first 10 keywords",
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ],
  "quality_notes": [],
  "risk_notes": []
}}
""".strip()


def title_repair_prompt(
    keywords: str,
    current_title: str,
    missing_keywords: List[str],
    hint: str,
) -> str:
    top_ten = keyword_list(keywords)[:TOP_KEYWORD_COUNT]

    return f"""
You are a senior Adobe Stock title editor.

TOP 10 KEYWORDS — all must appear verbatim in the new title:
{", ".join(top_ten)}

KEYWORDS CURRENTLY MISSING FROM THE TITLE:
{", ".join(missing_keywords) if missing_keywords else "None"}

CURRENT TITLE:
{current_title or "None"}

OPTIONAL CONTEXT:
{hint.strip() if hint.strip() else "None"}

Write one improved English stock title.

Mandatory rules:
- Include all 10 supplied keywords verbatim, ignoring capitalization.
- Keep the title natural, grammatical, readable, and human-written.
- Do not output a keyword list.
- Put the main subject and action early.
- Do not invent unsupported details.
- Prefer 100–200 characters.
- Never exceed 200 characters.
- Avoid excessive commas.
- Return a title meaningfully improved from the current title.

Return valid JSON only:
{{"title": "new title"}}
""".strip()


def quality_review_prompt(
    title: str,
    keywords: str,
) -> str:
    return f"""
You are a strict Adobe Stock metadata quality reviewer.

Review the supplied title and keywords against the uploaded image.

TITLE
{title}

KEYWORDS
{keywords}

Evaluate:
1. relevance_score: How accurately the title describes the visible image.
2. naturalness_score: How natural, clear, and grammatical the title is.
3. keyword_accuracy_score: How relevant and non-misleading the keywords are.

Scoring:
- Integer from 0 to 10.
- 10 means excellent.
- Be strict. Unsupported concepts must reduce the score.

Also return up to 5 short actionable notes.

Do not rewrite the metadata.
Return valid JSON only:

{{
  "relevance_score": 0,
  "naturalness_score": 0,
  "keyword_accuracy_score": 0,
  "notes": []
}}
""".strip()


# =========================================================
# 10) AI OPERATIONS
# =========================================================
def generate_metadata(
    image_bytes: bytes,
    api_key: str,
    model: str,
    category_name: str,
    category_num: int,
    hint: str,
    blacklist: List[str],
) -> Dict[str, Any]:
    try:
        client = OpenAI(api_key=api_key)

        response = call_openai(
            client=client,
            model=model,
            input_payload=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": metadata_prompt(
                            category_name,
                            category_num,
                            hint,
                            blacklist,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image_bytes),
                        "detail": "high",
                    },
                ],
            }],
            max_output_tokens=1800,
        )

        raw = (response.output_text or "").strip()
        parsed = extract_json(raw)

        title = normalize_title(parsed.get("title", ""))
        keywords = normalize_keywords(
            parsed.get("keywords", []),
            blacklist,
        )

        if not title:
            raise ValueError("โมเดลไม่ได้สร้าง Title")

        if not keywords:
            raise ValueError("โมเดลไม่ได้สร้าง Keywords")

        # ตรวจและแก้ Title อัตโนมัติ เพื่อให้ Top 10 ครบ 10/10
        repair_history: List[Dict[str, Any]] = []

        for attempt in range(1, MAX_TITLE_REPAIR_ATTEMPTS + 1):
            coverage = top_keyword_coverage(title, keywords)

            if coverage["complete"] and len(title) <= TITLE_MAX_LENGTH:
                break

            repair_response = call_openai(
                client=client,
                model=model,
                input_payload=[{
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": title_repair_prompt(
                            keywords=keywords,
                            current_title=title,
                            missing_keywords=coverage["missing"],
                            hint=hint,
                        ),
                    }],
                }],
                max_output_tokens=450,
            )

            repaired_raw = (
                repair_response.output_text or ""
            ).strip()

            new_title = normalize_title(
                extract_json(repaired_raw).get("title", "")
            )

            if new_title:
                title = new_title

            repair_history.append({
                "attempt": attempt,
                "missing_before": coverage["missing"],
                "title_after": title,
            })

        return {
            "title": title,
            "keywords": keywords,
            "quality_notes": (
                parsed.get("quality_notes", [])[:3]
                if isinstance(parsed.get("quality_notes", []), list)
                else []
            ),
            "risk_notes": (
                parsed.get("risk_notes", [])[:5]
                if isinstance(parsed.get("risk_notes", []), list)
                else []
            ),
            "repair_history": repair_history,
            "raw": raw,
            "error": False,
            "error_message": "",
        }

    except Exception as error:
        message = friendly_api_error(error, model)

        return {
            "title": "",
            "keywords": "",
            "quality_notes": [],
            "risk_notes": [],
            "repair_history": [],
            "raw": message,
            "error": True,
            "error_message": message,
        }


def regenerate_title_until_valid(
    keywords: str,
    current_title: str,
    api_key: str,
    model: str,
    hint: str,
) -> Dict[str, Any]:
    if keyword_count(keywords) < TOP_KEYWORD_COUNT:
        return {
            "title": current_title,
            "attempts": 0,
            "error": True,
            "error_message": "ต้องมีอย่างน้อย 10 Keywords ก่อน",
        }

    try:
        client = OpenAI(api_key=api_key)
        title = normalize_title(current_title)
        attempts = 0

        for attempt in range(1, MAX_TITLE_REPAIR_ATTEMPTS + 1):
            coverage = top_keyword_coverage(title, keywords)

            if (
                coverage["complete"]
                and 0 < len(title) <= TITLE_MAX_LENGTH
                and attempt > 1
            ):
                break

            response = call_openai(
                client=client,
                model=model,
                input_payload=[{
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": title_repair_prompt(
                            keywords=keywords,
                            current_title=title,
                            missing_keywords=coverage["missing"],
                            hint=hint,
                        ),
                    }],
                }],
                max_output_tokens=450,
            )

            generated_title = normalize_title(
                extract_json(
                    response.output_text or ""
                ).get("title", "")
            )

            if generated_title:
                title = generated_title

            attempts = attempt

            final_coverage = top_keyword_coverage(
                title,
                keywords,
            )

            if (
                final_coverage["complete"]
                and len(title) <= TITLE_MAX_LENGTH
            ):
                break

        return {
            "title": title,
            "attempts": attempts,
            "error": False,
            "error_message": "",
        }

    except Exception as error:
        return {
            "title": current_title,
            "attempts": 0,
            "error": True,
            "error_message": friendly_api_error(error, model),
        }


def ai_quality_review(
    image_bytes: bytes,
    title: str,
    keywords: str,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    try:
        client = OpenAI(api_key=api_key)

        response = call_openai(
            client=client,
            model=model,
            input_payload=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": quality_review_prompt(
                            title,
                            keywords,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url(image_bytes),
                        "detail": "high",
                    },
                ],
            }],
            max_output_tokens=500,
        )

        parsed = extract_json(
            response.output_text or ""
        )

        def bounded_score(key: str) -> int:
            try:
                return max(
                    0,
                    min(10, int(parsed.get(key, 0))),
                )
            except Exception:
                return 0

        return {
            "relevance_score": bounded_score(
                "relevance_score"
            ),
            "naturalness_score": bounded_score(
                "naturalness_score"
            ),
            "keyword_accuracy_score": bounded_score(
                "keyword_accuracy_score"
            ),
            "notes": (
                parsed.get("notes", [])[:5]
                if isinstance(parsed.get("notes", []), list)
                else []
            ),
            "error": False,
            "error_message": "",
        }

    except Exception as error:
        return {
            "relevance_score": 0,
            "naturalness_score": 0,
            "keyword_accuracy_score": 0,
            "notes": [],
            "error": True,
            "error_message": friendly_api_error(error, model),
        }


# =========================================================
# 11) EXPORT HELPERS
# =========================================================
def export_jpeg(
    data: bytes,
    title: str,
    keywords: str,
    extension: str,
) -> bytes:
    temp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".jpeg" if extension == ".jpeg" else ".jpg",
            delete=False,
        ) as temporary_file:
            temporary_file.write(data)
            temp_path = temporary_file.name

        info = IPTCInfo(temp_path, force=True)
        info["object name"] = title.encode("utf-8")
        info["caption/abstract"] = title.encode("utf-8")
        info["keywords"] = [
            keyword.encode("utf-8")
            for keyword in keyword_list(keywords)
        ]
        info.save()

        return Path(temp_path).read_bytes()

    except Exception:
        return data

    finally:
        if temp_path:
            for path in (
                Path(temp_path),
                Path(temp_path + "~"),
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


def export_png(
    data: bytes,
    title: str,
    keywords: str,
) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            original_size = source.size
            original_has_alpha = (
                source.mode in ("RGBA", "LA")
                or (
                    source.mode == "P"
                    and "transparency" in source.info
                )
            )

            metadata = PngInfo()

            for key, value in source.info.items():
                if (
                    isinstance(value, str)
                    and key.lower()
                    not in {
                        "title",
                        "description",
                        "caption",
                        "keywords",
                    }
                ):
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
                save_options["icc_profile"] = (
                    source.info["icc_profile"]
                )

            if source.info.get("dpi"):
                save_options["dpi"] = source.info["dpi"]

            if (
                source.mode in ("P", "L", "RGB")
                and "transparency" in source.info
            ):
                save_options["transparency"] = (
                    source.info["transparency"]
                )

            output = io.BytesIO()
            source.save(output, **save_options)
            exported = output.getvalue()

        with Image.open(io.BytesIO(exported)) as check:
            exported_has_alpha = (
                check.mode in ("RGBA", "LA")
                or (
                    check.mode == "P"
                    and "transparency" in check.info
                )
            )

            if check.size != original_size:
                raise ValueError("PNG dimensions changed")

            if original_has_alpha and not exported_has_alpha:
                raise ValueError("PNG transparency was lost")

        return exported

    except Exception:
        return data


def export_image(
    data: bytes,
    extension: str,
    title: str,
    keywords: str,
) -> bytes:
    if extension in {".jpg", ".jpeg"}:
        return export_jpeg(
            data,
            title,
            keywords,
            extension,
        )

    if extension == ".png":
        return export_png(
            data,
            title,
            keywords,
        )

    return data


# =========================================================
# 12) SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    manual_api_key = st.text_input(
        "🔑 OpenAI API Key",
        value="",
        type="password",
        placeholder=(
            "ไม่ต้องกรอก หากตั้ง OPENAI_API_KEY แล้ว"
        ),
    )

    env_api_key = os.getenv(
        "OPENAI_API_KEY",
        "",
    ).strip()

    try:
        secret_api_key = str(
            st.secrets.get(
                "OPENAI_API_KEY",
                "",
            )
        ).strip()
    except Exception:
        secret_api_key = ""

    api_key = (
        manual_api_key.strip()
        or env_api_key
        or secret_api_key
    )

    if st.button(
        "🔄 รีเฟรชรายการโมเดล",
        use_container_width=True,
    ):
        st.session_state.model_refresh_version += 1
        st.rerun()

    available_models, model_error = (
        get_available_models(api_key)
    )

    if available_models:
        model_choice = st.selectbox(
            "🤖 Model ที่ API Key เข้าถึงได้",
            options=available_models,
            index=0,
            help=(
                "Models API บอกสิทธิ์เข้าถึง "
                "แต่บางโมเดลอาจไม่รองรับ input_image"
            ),
        )

        use_custom_model = st.checkbox(
            "กรอก Model ID เอง",
            value=False,
        )

        if use_custom_model:
            model_choice = st.text_input(
                "Custom Model ID",
                value=model_choice,
            ).strip()

    else:
        if model_error:
            st.warning(
                "ดึงรายการโมเดลไม่ได้: "
                + redact_secrets(model_error)
            )

        model_choice = st.text_input(
            "🤖 Model ID",
            value="gpt-4.1",
            help=(
                "ใส่ API Key แล้วกดรีเฟรช "
                "เพื่อดูโมเดลที่บัญชีเข้าถึงได้จริง"
            ),
        ).strip()

    category_name = st.selectbox(
        "📁 Adobe Category",
        options=list(CATEGORY_DICT.keys()),
        index=2,
    )
    category_num = CATEGORY_DICT[category_name]

    hint = st.text_area(
        "💡 Context Hint",
        placeholder=(
            "เช่น young woman stretching "
            "before tennis workout"
        ),
    )

    blacklist_raw = st.text_area(
        "🛡️ Blacklist Keywords",
        value=DEFAULT_BLACKLIST,
    )
    blacklist = parse_blacklist(blacklist_raw)

    auto_ai_review = st.checkbox(
        "ให้ AI ตรวจความตรงภาพหลังสร้างอัตโนมัติ",
        value=True,
        help=(
            "แม่นขึ้น แต่มี API cost เพิ่มอีกหนึ่ง request ต่อภาพ"
        ),
    )

    prevent_failed_export = st.checkbox(
        "ไม่ให้ Export ภาพที่ไม่ผ่านคุณภาพ",
        value=True,
    )

    st.divider()

    server_files, server_bytes = directory_stats(
        APP_CACHE_DIR
    )

    st.caption(
        f"Server cache: {server_files:,} files • "
        f"{human_size(server_bytes)}"
    )

    confirm_cleanup = st.checkbox(
        "ยืนยันการล้างแคชบนเซิร์ฟเวอร์",
        value=False,
    )

    if st.button(
        "🧹 ล้างแคชภาพทั้งหมดบนเซิร์ฟเวอร์",
        use_container_width=True,
        disabled=not confirm_cleanup,
    ):
        deleted_files, deleted_bytes = (
            clear_server_cache()
        )

        st.success(
            f"ล้างแล้ว {deleted_files:,} ไฟล์ • "
            f"คืนพื้นที่ {human_size(deleted_bytes)}"
        )


# =========================================================
# 13) MAIN UI
# =========================================================
try:
    if st.session_state.flash_message:
        st.success(st.session_state.flash_message)
        st.session_state.flash_message = ""

    uploader_key = (
        f"image_uploader_"
        f"{st.session_state.uploader_version}"
    )

    uploaded_files = st.file_uploader(
        "📸 อัปโหลดรูปภาพ JPG, JPEG หรือ PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=uploader_key,
    )

    payloads = (
        prepare_uploads(uploaded_files)
        if uploaded_files
        else []
    )

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        analyze_all_clicked = st.button(
            "🚀 วิเคราะห์และตรวจคุณภาพทั้งหมด",
            type="primary",
            use_container_width=True,
            disabled=not payloads,
        )

    with action_col2:
        clear_page_clicked = st.button(
            "🗑️ ล้างภาพออกจากหน้าอัปโหลด",
            use_container_width=True,
            disabled=not payloads,
            help=(
                "ล้างเฉพาะภาพและผลในหน้านี้ "
                "ไม่ลบแคชบนเซิร์ฟเวอร์"
            ),
        )

    if clear_page_clicked:
        clear_page_uploads()
        st.rerun()

    if analyze_all_clicked:
        if not api_key:
            st.error("กรุณาใส่ OpenAI API Key")
        elif not model_choice:
            st.error("กรุณาเลือกหรือกรอก Model ID")
        else:
            valid_payloads = [
                payload
                for payload in payloads
                if payload["valid"]
            ]

            progress = st.progress(0)
            status = st.empty()

            for index, payload in enumerate(
                valid_payloads,
                start=1,
            ):
                status.info(
                    f"กำลังสร้าง Metadata "
                    f"{index}/{len(valid_payloads)}: "
                    f"{payload['original_name']}"
                )

                generation_cache_key = stable_hash({
                    "file_hash": sha256_bytes(
                        payload["bytes"]
                    ),
                    "model": model_choice,
                    "category": category_num,
                    "hint": hint,
                    "blacklist": blacklist,
                    "prompt_version": 8,
                })

                if (
                    generation_cache_key
                    in st.session_state.analysis_cache
                ):
                    generated = (
                        st.session_state.analysis_cache[
                            generation_cache_key
                        ]
                    )
                else:
                    generated = generate_metadata(
                        image_bytes=payload["bytes"],
                        api_key=api_key,
                        model=model_choice,
                        category_name=category_name,
                        category_num=category_num,
                        hint=hint,
                        blacklist=blacklist,
                    )
                    st.session_state.analysis_cache[
                        generation_cache_key
                    ] = generated

                ai_review: Optional[Dict[str, Any]] = None

                if (
                    auto_ai_review
                    and not generated["error"]
                ):
                    status.info(
                        f"กำลังตรวจความตรงภาพ "
                        f"{index}/{len(valid_payloads)}: "
                        f"{payload['original_name']}"
                    )

                    review_cache_key = stable_hash({
                        "file_hash": sha256_bytes(
                            payload["bytes"]
                        ),
                        "title": generated["title"],
                        "keywords": generated["keywords"],
                        "model": model_choice,
                        "review_version": 3,
                    })

                    if (
                        review_cache_key
                        in st.session_state.quality_cache
                    ):
                        ai_review = (
                            st.session_state.quality_cache[
                                review_cache_key
                            ]
                        )
                    else:
                        ai_review = ai_quality_review(
                            image_bytes=payload["bytes"],
                            title=generated["title"],
                            keywords=generated["keywords"],
                            api_key=api_key,
                            model=model_choice,
                        )
                        st.session_state.quality_cache[
                            review_cache_key
                        ] = ai_review

                quality = deterministic_quality_check(
                    title=generated["title"],
                    keywords=generated["keywords"],
                    blacklist=blacklist,
                    ai_review=(
                        ai_review
                        if ai_review
                        and not ai_review.get("error")
                        else None
                    ),
                )

                st.session_state.results[payload["id"]] = {
                    **payload,
                    **generated,
                    "category": category_num,
                    "ai_review": ai_review,
                    "quality": quality,
                }

                st.session_state[
                    f"title_{payload['id']}"
                ] = generated["title"]

                st.session_state[
                    f"keywords_{payload['id']}"
                ] = generated["keywords"]

                progress.progress(
                    index / max(len(valid_payloads), 1)
                )

            status.success(
                "สร้างและตรวจคุณภาพเรียบร้อย"
            )
            st.rerun()

    final_items: List[Dict[str, Any]] = []
    failed_items: List[str] = []

    for payload in payloads:
        with st.container(border=True):
            image_col, detail_col = st.columns(
                [1, 2],
                gap="large",
            )

            with image_col:
                st.image(
                    payload["bytes"],
                    use_container_width=True,
                )
                st.caption(payload["original_name"])

                if payload["valid"]:
                    info = payload["image_info"]

                    st.caption(
                        f"{info['width']:,} × "
                        f"{info['height']:,} px • "
                        f"{info['mode']} • "
                        f"{human_size(payload['size'])}"
                    )

                    if info["has_transparency"]:
                        st.success(
                            "PNG โปร่งใส — Export คง Alpha"
                        )
                else:
                    st.error(
                        payload["validation_error"]
                    )

            with detail_col:
                if not payload["valid"]:
                    continue

                file_id = payload["id"]
                result = st.session_state.results.get(
                    file_id
                )

                if not result:
                    st.info(
                        "ยังไม่ได้วิเคราะห์ภาพนี้"
                    )
                    continue

                if result.get("error"):
                    st.error(
                        redact_secrets(
                            result.get(
                                "error_message",
                                "เกิดข้อผิดพลาด",
                            )
                        )
                    )

                    with st.expander("Error detail"):
                        st.code(
                            redact_secrets(
                                result.get("raw", "")
                            )
                        )

                title_key = f"title_{file_id}"
                keywords_key = f"keywords_{file_id}"

                if title_key not in st.session_state:
                    st.session_state[title_key] = (
                        result.get("title", "")
                    )

                if keywords_key not in st.session_state:
                    st.session_state[keywords_key] = (
                        result.get("keywords", "")
                    )

                edited_title = st.text_area(
                    "Title",
                    key=title_key,
                    height=90,
                )

                edited_keywords = st.text_area(
                    "Keywords",
                    key=keywords_key,
                    height=150,
                )

                cleaned_title = normalize_title(
                    edited_title
                )
                cleaned_keywords = normalize_keywords(
                    edited_keywords,
                    blacklist=[],
                    limit=999,
                )

                st.session_state.results[file_id][
                    "title"
                ] = cleaned_title

                st.session_state.results[file_id][
                    "keywords"
                ] = cleaned_keywords

                button_col1, button_col2, button_col3 = (
                    st.columns(3)
                )

                with button_col1:
                    regenerate_clicked = st.button(
                        "✨ สร้าง Title ใหม่",
                        key=f"regenerate_{file_id}",
                        use_container_width=True,
                    )

                with button_col2:
                    quality_clicked = st.button(
                        "✅ ตรวจคุณภาพใหม่",
                        key=f"quality_{file_id}",
                        use_container_width=True,
                    )

                with button_col3:
                    delete_result_clicked = st.button(
                        "🗑️ ลบผลภาพนี้",
                        key=f"delete_{file_id}",
                        use_container_width=True,
                    )

                if regenerate_clicked:
                    if not api_key:
                        st.error(
                            "กรุณาใส่ OpenAI API Key"
                        )
                    else:
                        with st.spinner(
                            "กำลังสร้างและตรวจ Title ใหม่..."
                        ):
                            regenerated = (
                                regenerate_title_until_valid(
                                    keywords=cleaned_keywords,
                                    current_title=cleaned_title,
                                    api_key=api_key,
                                    model=model_choice,
                                    hint=hint,
                                )
                            )

                        if regenerated["error"]:
                            st.error(
                                redact_secrets(
                                    regenerated[
                                        "error_message"
                                    ]
                                )
                            )
                        else:
                            new_title = regenerated["title"]

                            st.session_state.results[
                                file_id
                            ]["title"] = new_title

                            st.session_state[
                                title_key
                            ] = new_title

                            # AI review เก่าหมดอายุ
                            st.session_state.results[
                                file_id
                            ]["ai_review"] = None

                            st.rerun()

                if quality_clicked:
                    ai_review = None

                    if not api_key:
                        st.warning(
                            "ไม่ได้ใส่ API Key: "
                            "จะตรวจเฉพาะกฎพื้นฐาน"
                        )
                    else:
                        with st.spinner(
                            "กำลังตรวจความตรงกับภาพ..."
                        ):
                            ai_review = ai_quality_review(
                                image_bytes=payload["bytes"],
                                title=cleaned_title,
                                keywords=cleaned_keywords,
                                api_key=api_key,
                                model=model_choice,
                            )

                        if ai_review["error"]:
                            st.error(
                                redact_secrets(
                                    ai_review[
                                        "error_message"
                                    ]
                                )
                            )
                            ai_review = None

                    quality = deterministic_quality_check(
                        title=cleaned_title,
                        keywords=cleaned_keywords,
                        blacklist=blacklist,
                        ai_review=ai_review,
                    )

                    st.session_state.results[file_id][
                        "ai_review"
                    ] = ai_review

                    st.session_state.results[file_id][
                        "quality"
                    ] = quality

                    st.rerun()

                if delete_result_clicked:
                    st.session_state.results.pop(
                        file_id,
                        None,
                    )
                    st.session_state.pop(
                        title_key,
                        None,
                    )
                    st.session_state.pop(
                        keywords_key,
                        None,
                    )
                    st.rerun()

                # คำนวณใหม่ทันทีทุกครั้งที่ผู้ใช้แก้ข้อความ
                ai_review = result.get("ai_review")

                quality = deterministic_quality_check(
                    title=cleaned_title,
                    keywords=cleaned_keywords,
                    blacklist=blacklist,
                    ai_review=(
                        ai_review
                        if ai_review
                        and not ai_review.get("error")
                        else None
                    ),
                )

                st.session_state.results[file_id][
                    "quality"
                ] = quality

                metric1, metric2, metric3, metric4 = (
                    st.columns(4)
                )

                metric1.metric(
                    "Quality Score",
                    f"{quality['score']}/100",
                )
                metric2.metric(
                    "Keywords",
                    f"{quality['keyword_count']}/49",
                )
                metric3.metric(
                    "Top 10 ใน Title",
                    f"{quality['coverage']['count']}/10",
                )
                metric4.metric(
                    "Title Length",
                    f"{quality['title_length']}/200",
                )

                if quality["level"] == "success":
                    st.success(quality["status"])
                elif quality["level"] == "warning":
                    st.warning(quality["status"])
                else:
                    st.error(quality["status"])

                with st.expander(
                    "🔎 รายละเอียดการตรวจคุณภาพ",
                    expanded=(
                        quality["level"] != "success"
                    ),
                ):
                    st.markdown("**10 Keywords แรก**")
                    st.write(
                        ", ".join(
                            quality["coverage"][
                                "top_ten"
                            ]
                        )
                        or "-"
                    )

                    st.markdown(
                        "**พบใน Title แบบตรงคำ**"
                    )
                    st.write(
                        ", ".join(
                            quality["coverage"][
                                "found"
                            ]
                        )
                        or "-"
                    )

                    st.markdown(
                        "**คำที่ยังขาดจาก Title**"
                    )
                    st.write(
                        ", ".join(
                            quality["coverage"][
                                "missing"
                            ]
                        )
                        or "ไม่มี — ครบ 10/10"
                    )

                    st.markdown("**คะแนนแยกส่วน**")
                    breakdown_df = pd.DataFrame([
                        {
                            "หัวข้อ": key,
                            "คะแนน": value,
                        }
                        for key, value in (
                            quality[
                                "score_breakdown"
                            ].items()
                        )
                    ])

                    st.dataframe(
                        breakdown_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if quality["issues"]:
                        st.markdown(
                            "**สิ่งที่ต้องแก้**"
                        )
                        for issue in quality["issues"]:
                            st.write(f"• {issue}")

                    if (
                        quality["ai_relevance"]
                        is not None
                    ):
                        st.markdown(
                            "**AI Quality Review**"
                        )
                        st.write(
                            "ความตรงภาพ: "
                            f"{quality['ai_relevance']}/10"
                        )
                        st.write(
                            "ความเป็นธรรมชาติ: "
                            f"{quality['ai_naturalness']}/10"
                        )
                        st.write(
                            "ความแม่นยำ Keywords: "
                            f"{quality['ai_keyword_accuracy']}/10"
                        )

                        for note in quality["ai_notes"]:
                            st.write(f"• {note}")
                    else:
                        st.info(
                            "ยังไม่มี AI Review "
                            "กด “ตรวจคุณภาพใหม่” "
                            "เพื่อตรวจความตรงกับภาพ"
                        )

                export_allowed = (
                    quality["level"] == "success"
                    or not prevent_failed_export
                )

                if export_allowed:
                    final_items.append({
                        "File ID": file_id,
                        "Filename": payload["safe_name"],
                        "Title": cleaned_title,
                        "Keywords": cleaned_keywords,
                        "Category": category_num,
                        "Releases": "",
                        "Quality Score": quality["score"],
                        "Quality Status": quality["status"],
                        "Top 10 Coverage": (
                            quality["coverage"]["count"]
                        ),
                        "bytes": payload["bytes"],
                        "extension": payload["extension"],
                    })
                else:
                    failed_items.append(
                        payload["original_name"]
                    )

    # =====================================================
    # 14) EXPORT
    # =====================================================
    if failed_items and prevent_failed_export:
        st.warning(
            "ภาพที่ยังไม่ผ่านและถูกกันออกจาก Export: "
            + ", ".join(failed_items)
        )

    if final_items:
        st.divider()
        st.subheader("📦 Export")

        adobe_df = pd.DataFrame(final_items)[[
            "Filename",
            "Title",
            "Keywords",
            "Category",
            "Releases",
        ]]

        quality_df = pd.DataFrame(final_items)[[
            "Filename",
            "Quality Score",
            "Quality Status",
            "Top 10 Coverage",
        ]]

        csv_bytes = adobe_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        quality_csv_bytes = quality_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "📊 Download Adobe CSV",
                data=csv_bytes,
                file_name="adobe_stock_metadata.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            create_zip_clicked = st.button(
                "📦 สร้าง ZIP พร้อม Metadata",
                use_container_width=True,
            )

        if create_zip_clicked:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer,
                "w",
                zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr(
                    "adobe_stock_metadata.csv",
                    csv_bytes,
                )
                archive.writestr(
                    "metadata_quality_report.csv",
                    quality_csv_bytes,
                )

                used_names: Dict[str, int] = {}

                for item in final_items:
                    filename = item["Filename"]
                    filename_key = filename.lower()
                    occurrence = (
                        used_names.get(
                            filename_key,
                            0,
                        )
                        + 1
                    )
                    used_names[filename_key] = occurrence

                    if occurrence > 1:
                        path = Path(filename)
                        filename = (
                            f"{path.stem}_{occurrence}"
                            f"{path.suffix}"
                        )

                    exported = export_image(
                        data=item["bytes"],
                        extension=item["extension"],
                        title=item["Title"],
                        keywords=item["Keywords"],
                    )

                    archive.writestr(
                        filename,
                        exported,
                    )

            zip_bytes = zip_buffer.getvalue()
            export_name = (
                f"adobe_stock_package_"
                f"{int(time.time())}.zip"
            )

            export_path = (
                EXPORT_CACHE_DIR / export_name
            )
            export_path.write_bytes(zip_bytes)

            st.session_state.generated_zip = (
                zip_bytes
            )
            st.session_state.generated_zip_name = (
                export_name
            )

        if st.session_state.generated_zip:
            st.download_button(
                "📂 Download ZIP",
                data=st.session_state.generated_zip,
                file_name=(
                    st.session_state
                    .generated_zip_name
                ),
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

        with st.expander("ดู Adobe CSV"):
            st.dataframe(
                adobe_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "ดู Quality Report"
        ):
            st.dataframe(
                quality_df,
                use_container_width=True,
                hide_index=True,
            )

except Exception:
    st.error("Application Error")

    with st.expander(
        "รายละเอียด Error",
        expanded=True,
    ):
        st.code(
            redact_secrets(
                traceback.format_exc()
            )
        )