import streamlit as st
import pandas as pd

from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

from iptcinfo3 import IPTCInfo
from openai import OpenAI

import base64
import hashlib
import io
import json
import os
import re
import tempfile
import time
import traceback
import zipfile

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


MODEL_OPTIONS = {
    "GPT-5.6 Terra — สมดุลคุณภาพและค่าใช้จ่าย": "gpt-5.6-terra",
    "GPT-5.6 Sol — คุณภาพสูงสุด": "gpt-5.6",
    "GPT-5.6 Luna — เร็วและประหยัด": "gpt-5.6-luna",
    "Custom — กรอก Model ID เอง": "custom",
}


DEFAULT_BLACKLIST = (
    "nike, apple, adidas, disney, marvel, coca-cola, samsung, sony, "
    "tesla, iphone, ipad, macbook, logo, trademark, brand, celebrity"
)


KEYWORD_LIMIT = 49
TOP_KEYWORD_COUNT = 10

# ไม่จำเป็นต้องยัดทั้ง 10 คำตรงตัว แต่ควรครอบคลุมแกนสำคัญอย่างน้อย 5 คำ
MIN_TOP_KEYWORD_COVERAGE = 5

# Adobe Stock title รองรับได้สูงสุด 200 ตัวอักษร
TITLE_MIN_LENGTH = 70
TITLE_MAX_LENGTH = 200

# รูปที่ส่ง AI เท่านั้นจะถูกย่อ
ANALYSIS_MAX_SIDE = 1800
ANALYSIS_JPEG_QUALITY = 90

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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
}


for state_key, default_value in DEFAULT_SESSION_STATE.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value


# =========================================================
# 4) BASIC HELPERS
# =========================================================
def normalize_spaces(value: Any) -> str:
    """รวมช่องว่างซ้ำและตัดช่องว่างหัวท้าย"""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_title(title: Any) -> str:
    """ปรับ Title ให้สะอาดโดยไม่เปลี่ยนความหมาย"""
    cleaned = normalize_spaces(title)
    return cleaned.strip(" ,;:-")


def parse_blacklist(raw_text: str) -> List[str]:
    """แปลง blacklist จากข้อความคั่น comma เป็น list"""
    return [
        normalize_spaces(item).lower()
        for item in raw_text.split(",")
        if normalize_spaces(item)
    ]


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def get_original_extension(filename: str) -> str:
    """ดึง extension ต้นฉบับและ normalize เป็นตัวพิมพ์เล็ก"""
    extension = Path(filename).suffix.lower()

    if extension == ".jpe":
        extension = ".jpg"

    return extension


def sanitize_filename_preserve_extension(filename: str) -> str:
    """
    ทำชื่อไฟล์ให้ปลอดภัยและรักษา extension ต้นฉบับ
    เช่น product.png -> product.png
    """
    original_name = os.path.basename(filename)
    path = Path(original_name)

    extension = path.suffix.lower()
    stem = path.stem

    if extension not in SUPPORTED_EXTENSIONS:
        extension = ".png" if extension == ".png" else ".jpg"

    stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip(" ._-")

    if not stem:
        stem = "image"

    return f"{stem}{extension}"


def unique_file_id(filename: str, file_bytes: bytes) -> str:
    """สร้าง ID ที่ไม่ชนกัน แม้ชื่อไฟล์เหมือนกัน"""
    safe_name = sanitize_filename_preserve_extension(filename)
    return f"{safe_name}__{file_hash(file_bytes)[:16]}"


def make_unique_export_filenames(
    payloads: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    ป้องกันชื่อไฟล์ซ้ำใน ZIP และ CSV

    image.png
    image_2.png
    image_3.png
    """
    used_names: Dict[str, int] = {}
    export_names: Dict[str, str] = {}

    for payload in payloads:
        safe_name = payload["safe_name"]
        path = Path(safe_name)

        stem = path.stem
        extension = path.suffix.lower()

        normalized_key = safe_name.lower()
        current_count = used_names.get(normalized_key, 0) + 1
        used_names[normalized_key] = current_count

        if current_count == 1:
            final_name = safe_name
        else:
            final_name = f"{stem}_{current_count}{extension}"

            while final_name.lower() in used_names:
                current_count += 1
                final_name = f"{stem}_{current_count}{extension}"

            used_names[final_name.lower()] = 1

        export_names[payload["id"]] = final_name

    return export_names


# =========================================================
# 5) IMAGE VALIDATION AND PREPARATION
# =========================================================
def validate_image_bytes(
    file_bytes: bytes,
    filename: str,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    ตรวจสอบว่าไฟล์เป็นภาพจริง และอ่านข้อมูลพื้นฐาน
    """
    extension = get_original_extension(filename)

    if extension not in SUPPORTED_EXTENSIONS:
        return (
            False,
            f"ไม่รองรับนามสกุล {extension or 'ไม่ทราบชนิด'}",
            {},
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image.verify()

        with Image.open(io.BytesIO(file_bytes)) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format or ""

            has_transparency = (
                mode in ("RGBA", "LA")
                or (
                    mode == "P"
                    and "transparency" in image.info
                )
            )

        expected_formats = {
            ".jpg": {"JPEG"},
            ".jpeg": {"JPEG"},
            ".png": {"PNG"},
        }

        allowed_formats = expected_formats.get(extension, set())

        if image_format.upper() not in allowed_formats:
            return (
                False,
                (
                    f"นามสกุลไฟล์เป็น {extension} "
                    f"แต่ข้อมูลภายในเป็น {image_format or 'unknown'}"
                ),
                {},
            )

        return (
            True,
            None,
            {
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format.upper(),
                "has_transparency": has_transparency,
            },
        )

    except UnidentifiedImageError:
        return False, "ไฟล์นี้ไม่ใช่ภาพที่ PIL รองรับ", {}

    except Exception as error:
        return False, f"เปิดไฟล์ไม่ได้: {error}", {}


def prepare_uploaded_payloads(
    uploaded_files: List[Any],
) -> List[Dict[str, Any]]:
    """
    อ่านไฟล์อัปโหลดเพียงครั้งเดียวและเก็บ bytes ไว้ใน payload
    """
    payloads: List[Dict[str, Any]] = []
    seen_ids = set()

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        file_id = unique_file_id(uploaded_file.name, file_bytes)

        if file_id in seen_ids:
            continue

        seen_ids.add(file_id)

        valid, validation_error, image_info = validate_image_bytes(
            file_bytes=file_bytes,
            filename=uploaded_file.name,
        )

        payloads.append(
            {
                "id": file_id,
                "original_name": uploaded_file.name,
                "safe_name": sanitize_filename_preserve_extension(
                    uploaded_file.name
                ),
                "extension": get_original_extension(uploaded_file.name),
                "bytes": file_bytes,
                "size": len(file_bytes),
                "mime": getattr(uploaded_file, "type", ""),
                "valid": valid,
                "validation_error": validation_error,
                "image_info": image_info,
            }
        )

    return payloads


def optimize_image_for_analysis(image_bytes: bytes) -> bytes:
    """
    สร้างสำเนาขนาดเล็กสำหรับส่ง AI เท่านั้น

    ไฟล์ต้นฉบับสำหรับ Export จะไม่ถูกลดขนาด
    """
    with Image.open(io.BytesIO(image_bytes)) as source_image:
        # หมุนภาพตาม EXIF เพื่อให้ AI เห็นทิศทางที่ถูกต้อง
        image = ImageOps.exif_transpose(source_image)

        if image.mode in ("RGBA", "LA"):
            # สร้างพื้นหลังขาวเฉพาะสำเนาที่ส่ง AI
            background = Image.new("RGB", image.size, (255, 255, 255))

            if image.mode == "RGBA":
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)
            else:
                alpha = image.getchannel("A")
                background.paste(image.convert("RGB"), mask=alpha)

            image = background

        elif image.mode == "P":
            if "transparency" in image.info:
                rgba_image = image.convert("RGBA")
                background = Image.new(
                    "RGB",
                    rgba_image.size,
                    (255, 255, 255),
                )
                background.paste(
                    rgba_image.convert("RGB"),
                    mask=rgba_image.getchannel("A"),
                )
                image = background
            else:
                image = image.convert("RGB")

        elif image.mode != "RGB":
            image = image.convert("RGB")

        image.thumbnail(
            (ANALYSIS_MAX_SIDE, ANALYSIS_MAX_SIDE),
            Image.Resampling.LANCZOS,
        )

        output_buffer = io.BytesIO()

        image.save(
            output_buffer,
            format="JPEG",
            quality=ANALYSIS_JPEG_QUALITY,
            optimize=True,
            subsampling=0,
        )

        return output_buffer.getvalue()


# =========================================================
# 6) KEYWORD NORMALIZATION
# =========================================================
def split_keywords(raw_keywords: Any) -> List[str]:
    """
    รองรับทั้ง list และข้อความคั่น comma/newline
    """
    if isinstance(raw_keywords, list):
        return [
            normalize_spaces(keyword)
            for keyword in raw_keywords
            if normalize_spaces(keyword)
        ]

    if isinstance(raw_keywords, str):
        normalized = raw_keywords.replace("\n", ",")
        return [
            normalize_spaces(keyword)
            for keyword in normalized.split(",")
            if normalize_spaces(keyword)
        ]

    return []


def simple_keyword_root(keyword: str) -> str:
    """
    ลดการซ้ำแบบ singular/plural อย่างเบื้องต้น

    ไม่ตัด s หากคำสั้นเกินไป เพื่อไม่ให้เกิด false positive ง่ายเกินไป
    """
    word = keyword.lower().strip()

    if " " in word:
        return word

    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"

    if word.endswith("ves") and len(word) > 5:
        return word[:-3] + "f"

    if word.endswith("es") and len(word) > 5:
        return word[:-2]

    if word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
        return word[:-1]

    return word


def contains_blacklisted_term(
    text: str,
    blacklist_words: List[str],
) -> bool:
    """
    ตรวจ blacklist แบบคำหรือวลี
    """
    normalized_text = f" {re.sub(r'[^a-z0-9 ]+', ' ', text.lower())} "

    for blocked_word in blacklist_words:
        normalized_blocked = re.sub(
            r"[^a-z0-9 ]+",
            " ",
            blocked_word.lower(),
        ).strip()

        if not normalized_blocked:
            continue

        if f" {normalized_blocked} " in normalized_text:
            return True

    return False


def normalize_keywords(
    raw_keywords: Any,
    blacklist_words: List[str],
    limit: int = KEYWORD_LIMIT,
) -> str:
    """
    ทำความสะอาด keyword โดยรักษาลำดับความสำคัญเดิม
    """
    raw_list = split_keywords(raw_keywords)

    cleaned_keywords: List[str] = []
    seen_exact = set()
    seen_roots = set()

    forbidden_characters = {
        "#",
        "@",
        "\\",
        "|",
        "{",
        "}",
        "[",
        "]",
        "<",
        ">",
    }

    for keyword in raw_list:
        cleaned = normalize_spaces(keyword).strip(" ,;:.")

        if not cleaned:
            continue

        if len(cleaned) > 80:
            continue

        if any(char in cleaned for char in forbidden_characters):
            continue

        exact_key = cleaned.lower()
        root_key = simple_keyword_root(cleaned)

        if contains_blacklisted_term(cleaned, blacklist_words):
            continue

        if exact_key in seen_exact:
            continue

        # ลดเฉพาะคำเดี่ยว singular/plural ที่ซ้ำชัดเจน
        if " " not in cleaned and root_key in seen_roots:
            continue

        seen_exact.add(exact_key)

        if " " not in cleaned:
            seen_roots.add(root_key)

        cleaned_keywords.append(cleaned)

        if len(cleaned_keywords) >= limit:
            break

    return ", ".join(cleaned_keywords)


def get_keyword_list(keywords: str) -> List[str]:
    return [
        normalize_spaces(keyword)
        for keyword in keywords.split(",")
        if normalize_spaces(keyword)
    ]


def count_keywords(keywords: str) -> int:
    return len(get_keyword_list(keywords))


# =========================================================
# 7) TITLE AND KEYWORD VALIDATION
# =========================================================
def normalize_match_text(text: str) -> str:
    return re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text.lower(),
    ).strip()


def keyword_is_used_in_title(
    keyword: str,
    title: str,
) -> bool:
    """
    ตรวจว่าคำหรือแกนของ keyword อยู่ใน Title หรือไม่

    Phrase เช่น "active lifestyle"
    ต้องพบคำสำคัญของ phrase อย่างน้อย 2 คำ
    """
    normalized_title = normalize_match_text(title)
    normalized_keyword = normalize_match_text(keyword)

    if not normalized_title or not normalized_keyword:
        return False

    padded_title = f" {normalized_title} "
    padded_keyword = f" {normalized_keyword} "

    if padded_keyword in padded_title:
        return True

    keyword_parts = [
        part
        for part in normalized_keyword.split()
        if len(part) >= 3
    ]

    if not keyword_parts:
        return False

    if len(keyword_parts) == 1:
        root = simple_keyword_root(keyword_parts[0])

        title_parts = {
            simple_keyword_root(part)
            for part in normalized_title.split()
        }

        return root in title_parts

    required_matches = min(2, len(keyword_parts))

    title_parts = {
        simple_keyword_root(part)
        for part in normalized_title.split()
    }

    keyword_roots = [
        simple_keyword_root(part)
        for part in keyword_parts
    ]

    matches = sum(
        1
        for root in keyword_roots
        if root in title_parts
    )

    return matches >= required_matches


def top_keywords_used_in_title(
    title: str,
    keywords: str,
    top_n: int = TOP_KEYWORD_COUNT,
) -> List[str]:
    top_keywords = get_keyword_list(keywords)[:top_n]

    return [
        keyword
        for keyword in top_keywords
        if keyword_is_used_in_title(keyword, title)
    ]


def validate_title(
    title: str,
    keywords: str,
    blacklist_words: List[str],
) -> List[str]:
    errors: List[str] = []
    cleaned_title = normalize_title(title)

    if not cleaned_title:
        errors.append("Title ว่าง")
        return errors

    title_length = len(cleaned_title)

    if title_length < TITLE_MIN_LENGTH:
        errors.append(
            f"Title สั้นกว่า {TITLE_MIN_LENGTH} ตัวอักษร"
        )

    if title_length > TITLE_MAX_LENGTH:
        errors.append(
            f"Title ยาวเกิน {TITLE_MAX_LENGTH} ตัวอักษร"
        )

    if contains_blacklisted_term(cleaned_title, blacklist_words):
        errors.append("Title มีคำจาก Blacklist")

    comma_count = cleaned_title.count(",")

    if comma_count >= 4:
        errors.append(
            "Title มี comma มากเกินไปและอาจดูเป็น keyword dump"
        )

    matched_keywords = top_keywords_used_in_title(
        title=cleaned_title,
        keywords=keywords,
    )

    if len(matched_keywords) < MIN_TOP_KEYWORD_COVERAGE:
        errors.append(
            "Title ยังครอบคลุมแกนจาก 10 Keywords แรกน้อยเกินไป "
            f"({len(matched_keywords)}/{TOP_KEYWORD_COUNT})"
        )

    return errors


def validate_keywords(
    keywords: str,
    blacklist_words: List[str],
) -> List[str]:
    errors: List[str] = []
    keyword_list = get_keyword_list(keywords)

    if len(keyword_list) != KEYWORD_LIMIT:
        errors.append(
            f"มี Keywords {len(keyword_list)} คำ "
            f"ต้องการ {KEYWORD_LIMIT} คำ"
        )

    lower_keywords = [
        keyword.lower()
        for keyword in keyword_list
    ]

    duplicate_keywords = sorted(
        {
            keyword
            for keyword in lower_keywords
            if lower_keywords.count(keyword) > 1
        }
    )

    if duplicate_keywords:
        errors.append(
            "มี Keyword ซ้ำ: "
            + ", ".join(duplicate_keywords[:10])
        )

    blocked_keywords = [
        keyword
        for keyword in keyword_list
        if contains_blacklisted_term(
            keyword,
            blacklist_words,
        )
    ]

    if blocked_keywords:
        errors.append(
            "มี Keyword จาก Blacklist: "
            + ", ".join(blocked_keywords[:10])
        )

    return errors


def calculate_quality_score(
    title: str,
    keywords: str,
    blacklist_words: List[str],
) -> int:
    score = 100

    title_errors = validate_title(
        title,
        keywords,
        blacklist_words,
    )

    keyword_errors = validate_keywords(
        keywords,
        blacklist_words,
    )

    score -= len(title_errors) * 12
    score -= len(keyword_errors) * 15

    keyword_count = count_keywords(keywords)

    if keyword_count < KEYWORD_LIMIT:
        score -= min(
            20,
            KEYWORD_LIMIT - keyword_count,
        )

    coverage = len(
        top_keywords_used_in_title(
            title=title,
            keywords=keywords,
        )
    )

    if coverage < MIN_TOP_KEYWORD_COVERAGE:
        score -= (
            MIN_TOP_KEYWORD_COVERAGE - coverage
        ) * 4

    return max(0, min(100, score))


# =========================================================
# 8) OPENAI PROMPTS
# =========================================================
def build_metadata_prompt(
    category_name: str,
    category_num: int,
    hint: str,
    blacklist_words: List[str],
    title_style: str,
    keyword_style: str,
) -> str:
    blacklist_text = (
        ", ".join(blacklist_words)
        if blacklist_words
        else "None"
    )

    return f"""
You are a professional Adobe Stock contributor and metadata editor.

Analyze the uploaded image and generate commercially useful Adobe Stock metadata.

USER CONTEXT
- Context hint: {hint.strip() if hint.strip() else "None"}
- Selected Adobe category: {category_name}
- Category ID: {category_num}
- Preferred title style: {title_style}
- Preferred keyword strategy: {keyword_style}
- Forbidden words: {blacklist_text}

WORKFLOW — FOLLOW THIS ORDER

STEP 1: Analyze the visible image accurately.
Do not invent people, objects, locations, emotions, activities, seasons,
cultures, technologies, brands, or concepts that are not reasonably visible.

STEP 2: Generate exactly 49 English keywords.
The first 10 keywords are the most important search terms.
Order all keywords from most important to least important.

The first 10 should collectively describe:
- the primary subject
- the primary action or condition
- the environment or location
- the strongest commercial concept
- important visual attributes when relevant

STEP 3: Create the English title AFTER the keywords are finalized.
Use the meaning and strongest search intent of the first 10 keywords
as the foundation of the title.

TITLE REQUIREMENTS
- Exactly one English title.
- Between 100 and 200 characters when reasonably possible.
- Clear, natural, descriptive, fluent, and easy to understand.
- Written like a human stock contributor.
- Naturally incorporate as many important first-10 keywords as useful.
- Do not mechanically paste all keywords into one sentence.
- Do not create a comma-separated keyword list.
- Do not repeat the same idea unnaturally.
- Put the main subject and action early in the title.
- Do not start with “Image of”, “Photo of”, or “Picture of”.
- No promotional claims.
- No brands, logos, trademarks, celebrity names, copyrighted characters,
  or unsupported proper nouns.

KEYWORD REQUIREMENTS
- Exactly 49 English keywords.
- Comma is not needed because the output must be a JSON array.
- Most important keywords first.
- Use concise stock-search language.
- Include useful multi-word phrases only when buyers are likely to search them.
- Avoid irrelevant concepts and keyword spam.
- Avoid unnecessary singular/plural duplication.
- No brands, trademarks, copyrighted names, or celebrity names.

OUTPUT
Return only valid JSON using this exact structure:

{{
  "title": "Natural English stock title",
  "keywords": [
    "keyword 1",
    "keyword 2",
    "keyword 3"
  ],
  "quality_notes": [],
  "risk_notes": []
}}

quality_notes:
- Maximum 3 short English notes.
- Mention metadata concerns only when useful.

risk_notes:
- Mention a possible model release, property release,
  visible logo, trademark, or intellectual property risk only if relevant.
- Otherwise return an empty array.

Return JSON only.
""".strip()


def build_title_only_prompt(
    keywords: str,
    current_title: str,
    hint: str,
    title_style: str,
    blacklist_words: List[str],
) -> str:
    keyword_list = get_keyword_list(keywords)
    top_keywords = keyword_list[:TOP_KEYWORD_COUNT]

    blacklist_text = (
        ", ".join(blacklist_words)
        if blacklist_words
        else "None"
    )

    top_keyword_text = ", ".join(top_keywords)

    return f"""
You are a professional Adobe Stock title editor.

Create a NEW English stock title based on the current metadata.

TOP 10 KEYWORDS IN PRIORITY ORDER
{top_keyword_text}

OPTIONAL CONTEXT
{hint.strip() if hint.strip() else "None"}

CURRENT TITLE TO IMPROVE OR REPLACE
{current_title.strip() if current_title.strip() else "None"}

TITLE STYLE
{title_style}

FORBIDDEN WORDS
{blacklist_text}

CRITICAL RULES
- Create exactly one new English title.
- Use the meaning and search intent of the 10 keywords as the foundation.
- Prioritize the earliest keywords.
- Naturally use as many relevant top keywords as possible.
- Do not simply paste the keywords together.
- The title must sound natural, clear, descriptive, and human-written.
- Keep the primary subject and action near the beginning.
- Target approximately 100 to 200 characters.
- Maximum 200 characters.
- Avoid keyword stuffing.
- Avoid unnecessary commas.
- Do not start with “Image of”, “Photo of”, or “Picture of”.
- Do not add details unsupported by the keywords or context.
- Do not use brands, trademarks, logos, celebrities,
  copyrighted characters, or forbidden words.
- The new title should differ meaningfully from the current title.

Return only valid JSON:

{{
  "title": "New natural English stock title"
}}

Return JSON only.
""".strip()


# =========================================================
# 9) OPENAI RESPONSE HELPERS
# =========================================================
def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """
    รองรับทั้ง JSON ตรง ๆ และ JSON ที่ติด code fence มา
    """
    text = (raw_text or "").strip()

    if not text:
        raise ValueError("Model returned empty output")

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError("ไม่พบ JSON object ในคำตอบของโมเดล")

    parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("JSON output ไม่ใช่ object")

    return parsed


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    input_payload: List[Dict[str, Any]],
    max_output_tokens: int,
    retries: int = 3,
) -> Any:
    """
    Retry แบบ exponential backoff อย่างง่าย
    """
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

    if last_error:
        raise last_error

    raise RuntimeError("OpenAI request failed")


def repair_json_output(
    client: OpenAI,
    model: str,
    raw_text: str,
    output_type: str,
) -> Dict[str, Any]:
    """
    ซ่อมรูปแบบ JSON เมื่อโมเดลตอบคลาดเคลื่อน
    """
    if output_type == "title":
        required_schema = """
{
  "title": "string"
}
""".strip()
    else:
        required_schema = """
{
  "title": "string",
  "keywords": ["exactly 49 strings"],
  "quality_notes": [],
  "risk_notes": []
}
""".strip()

    repair_prompt = f"""
Convert the content below into valid JSON.

Required structure:

{required_schema}

Do not explain.
Do not use Markdown.
Return JSON only.

CONTENT TO REPAIR:
{raw_text}
""".strip()

    response = call_openai_with_retry(
        client=client,
        model=model,
        input_payload=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": repair_prompt,
                    }
                ],
            }
        ],
        max_output_tokens=900,
        retries=2,
    )

    repaired_text = (
        getattr(response, "output_text", "") or ""
    ).strip()

    return extract_json_object(repaired_text)


# =========================================================
# 10) AI ANALYSIS
# =========================================================
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

        analysis_bytes = optimize_image_for_analysis(
            image_bytes
        )

        base64_image = base64.b64encode(
            analysis_bytes
        ).decode("utf-8")

        prompt = build_metadata_prompt(
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
                        {
                            "type": "input_text",
                            "text": prompt,
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                "data:image/jpeg;base64,"
                                f"{base64_image}"
                            ),
                            "detail": "high",
                        },
                    ],
                }
            ],
            max_output_tokens=1400,
            retries=3,
        )

        raw_text = (
            getattr(response, "output_text", "") or ""
        ).strip()

        try:
            parsed_data = extract_json_object(raw_text)

        except Exception:
            parsed_data = repair_json_output(
                client=client,
                model=model,
                raw_text=raw_text,
                output_type="metadata",
            )

        title = normalize_title(
            parsed_data.get("title", "")
        )

        keywords = normalize_keywords(
            parsed_data.get("keywords", []),
            blacklist_words=blacklist_words,
            limit=KEYWORD_LIMIT,
        )

        quality_notes = parsed_data.get(
            "quality_notes",
            [],
        )

        risk_notes = parsed_data.get(
            "risk_notes",
            [],
        )

        if not isinstance(quality_notes, list):
            quality_notes = []

        if not isinstance(risk_notes, list):
            risk_notes = []

        error_messages: List[str] = []

        if not title:
            error_messages.append("โมเดลไม่ได้สร้าง Title")

        if not keywords:
            error_messages.append("โมเดลไม่ได้สร้าง Keywords")

        return {
            "title": title,
            "keywords": keywords,
            "quality_notes": [
                normalize_spaces(note)
                for note in quality_notes[:3]
                if normalize_spaces(note)
            ],
            "risk_notes": [
                normalize_spaces(note)
                for note in risk_notes[:5]
                if normalize_spaces(note)
            ],
            "raw": raw_text,
            "error": bool(error_messages),
            "error_message": "; ".join(error_messages),
        }

    except Exception as error:
        return {
            "title": "",
            "keywords": "",
            "quality_notes": [],
            "risk_notes": [],
            "raw": (
                f"{type(error).__name__}: {error}"
            ),
            "error": True,
            "error_message": (
                f"{type(error).__name__}: {error}"
            ),
        }


def regenerate_title_with_openai(
    keywords: str,
    current_title: str,
    hint: str,
    api_key: str,
    model: str,
    title_style: str,
    blacklist_words: List[str],
) -> Dict[str, Any]:
    """
    สร้างเฉพาะ Title ใหม่จาก Keywords ปัจจุบัน
    ไม่วิเคราะห์ภาพและไม่เปลี่ยน Keywords
    """
    keyword_list = get_keyword_list(keywords)

    if len(keyword_list) < TOP_KEYWORD_COUNT:
        return {
            "title": current_title,
            "raw": "",
            "error": True,
            "error_message": (
                "ต้องมีอย่างน้อย 10 Keywords "
                "ก่อนสร้าง Title ใหม่"
            ),
        }

    try:
        client = OpenAI(api_key=api_key)

        prompt = build_title_only_prompt(
            keywords=keywords,
            current_title=current_title,
            hint=hint,
            title_style=title_style,
            blacklist_words=blacklist_words,
        )

        response = call_openai_with_retry(
            client=client,
            model=model,
            input_payload=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            max_output_tokens=400,
            retries=3,
        )

        raw_text = (
            getattr(response, "output_text", "") or ""
        ).strip()

        try:
            parsed_data = extract_json_object(raw_text)

        except Exception:
            parsed_data = repair_json_output(
                client=client,
                model=model,
                raw_text=raw_text,
                output_type="title",
            )

        new_title = normalize_title(
            parsed_data.get("title", "")
        )

        if not new_title:
            raise ValueError(
                "โมเดลไม่ได้สร้าง Title ใหม่"
            )

        if len(new_title) > TITLE_MAX_LENGTH:
            # ไม่ตัดคำกลางประโยคแบบดื้อ ๆ
            shortened = new_title[:TITLE_MAX_LENGTH]
            last_space = shortened.rfind(" ")

            if last_space > 100:
                new_title = shortened[:last_space].rstrip(" ,;:-.")
            else:
                new_title = shortened.rstrip(" ,;:-.")

        return {
            "title": new_title,
            "raw": raw_text,
            "error": False,
            "error_message": "",
        }

    except Exception as error:
        return {
            "title": current_title,
            "raw": (
                f"{type(error).__name__}: {error}"
            ),
            "error": True,
            "error_message": (
                f"{type(error).__name__}: {error}"
            ),
        }


# =========================================================
# 11) CACHE HELPERS
# =========================================================
def make_analysis_cache_key(
    image_bytes: bytes,
    model: str,
    category_num: int,
    hint: str,
    blacklist_words: List[str],
    title_style: str,
    keyword_style: str,
) -> str:
    settings_text = json.dumps(
        {
            "model": model,
            "category_num": category_num,
            "hint": hint,
            "blacklist": sorted(blacklist_words),
            "title_style": title_style,
            "keyword_style": keyword_style,
            "prompt_version": 4,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    combined = (
        image_bytes
        + settings_text.encode("utf-8")
    )

    return hashlib.sha256(combined).hexdigest()


def make_title_cache_key(
    keywords: str,
    current_title: str,
    model: str,
    hint: str,
    title_style: str,
    blacklist_words: List[str],
) -> str:
    settings_text = json.dumps(
        {
            "keywords": keywords,
            "current_title": current_title,
            "model": model,
            "hint": hint,
            "title_style": title_style,
            "blacklist": sorted(blacklist_words),
            "prompt_version": 4,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        settings_text.encode("utf-8")
    ).hexdigest()


# =========================================================
# 12) RESULT MANAGEMENT
# =========================================================
def save_analysis_result(
    payload: Dict[str, Any],
    result: Dict[str, Any],
    category_num: int,
) -> None:
    file_id = payload["id"]

    st.session_state.results[file_id] = {
        "id": file_id,
        "original_name": payload["original_name"],
        "safe_name": payload["safe_name"],
        "extension": payload["extension"],
        "bytes": payload["bytes"],
        "size": payload["size"],
        "mime": payload["mime"],
        "image_info": payload["image_info"],
        "title": result.get("title", ""),
        "keywords": result.get("keywords", ""),
        "category_id": category_num,
        "quality_notes": result.get(
            "quality_notes",
            [],
        ),
        "risk_notes": result.get(
            "risk_notes",
            [],
        ),
        "raw": result.get("raw", ""),
        "error": result.get("error", False),
        "error_message": result.get(
            "error_message",
            "",
        ),
    }

    # อัปเดต widget state ให้เห็นค่าทันที
    st.session_state[f"title_{file_id}"] = result.get(
        "title",
        "",
    )

    st.session_state[f"keywords_{file_id}"] = result.get(
        "keywords",
        "",
    )


def sync_widget_values_to_result(file_id: str) -> None:
    """
    นำค่าที่ผู้ใช้แก้ในช่อง Title/Keywords กลับเข้า results
    """
    if file_id not in st.session_state.results:
        return

    title_key = f"title_{file_id}"
    keywords_key = f"keywords_{file_id}"

    if title_key in st.session_state:
        st.session_state.results[file_id]["title"] = (
            normalize_title(
                st.session_state[title_key]
            )
        )

    if keywords_key in st.session_state:
        st.session_state.results[file_id]["keywords"] = (
            st.session_state[keywords_key]
        )


def remove_result_widget_state(file_id: str) -> None:
    widget_keys = [
        f"title_{file_id}",
        f"keywords_{file_id}",
    ]

    for widget_key in widget_keys:
        st.session_state.pop(
            widget_key,
            None,
        )


def clear_all_uploaded_images() -> None:
    """
    ล้างภาพทั้งหมดโดยเปลี่ยน key ของ file_uploader
    """
    for file_id in list(
        st.session_state.results.keys()
    ):
        remove_result_widget_state(file_id)

    st.session_state.results = {}
    st.session_state.generated_zip = None
    st.session_state.generated_zip_name = ""

    st.session_state.uploader_version += 1


# =========================================================
# 13) METADATA EXPORT
# =========================================================
def export_jpeg_with_iptc(
    original_bytes: bytes,
    title: str,
    keywords: str,
    original_extension: str,
) -> bytes:
    """
    ฝัง IPTC ลง JPG/JPEG โดยใช้ไฟล์ต้นฉบับโดยตรง

    IPTCInfo แก้ metadata ในไฟล์ JPEG
    โดยไม่เรียก PIL save เพื่อหลีกเลี่ยงการ re-encode ภาพ
    """
    suffix = (
        ".jpeg"
        if original_extension == ".jpeg"
        else ".jpg"
    )

    temp_path: Optional[str] = None
    backup_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_file.write(original_bytes)
            temp_path = temporary_file.name

        info = IPTCInfo(
            temp_path,
            force=True,
        )

        info["object name"] = title.encode("utf-8")
        info["caption/abstract"] = title.encode("utf-8")

        info["keywords"] = [
            keyword.encode("utf-8")
            for keyword in get_keyword_list(keywords)
        ]

        info.save()

        backup_path = f"{temp_path}~"

        with open(temp_path, "rb") as saved_file:
            return saved_file.read()

    except Exception:
        # หากฝัง IPTC ไม่สำเร็จ คืนต้นฉบับโดยไม่ลดคุณภาพ
        return original_bytes

    finally:
        for path_to_remove in [
            temp_path,
            backup_path,
        ]:
            if (
                path_to_remove
                and os.path.exists(path_to_remove)
            ):
                try:
                    os.remove(path_to_remove)
                except OSError:
                    pass


def export_png_with_metadata(
    original_bytes: bytes,
    title: str,
    keywords: str,
) -> bytes:
    """
    PNG เป็น lossless และรักษา alpha channel

    ฟังก์ชันนี้:
    - ไม่ resize
    - ไม่ flatten transparency
    - ไม่ convert RGBA เป็น RGB
    - เก็บ ICC profile และ DPI เท่าที่ Pillow อ่านได้
    - เพิ่ม Title, Description และ Keywords ใน PNG text chunks
    """
    try:
        with Image.open(
            io.BytesIO(original_bytes)
        ) as source_image:
            source_image.load()

            original_mode = source_image.mode
            original_size = source_image.size

            png_metadata = PngInfo()

            # เก็บ text metadata เดิมที่เป็น string
            for metadata_key, metadata_value in (
                source_image.info.items()
            ):
                if (
                    isinstance(metadata_value, str)
                    and metadata_key.lower()
                    not in {
                        "title",
                        "description",
                        "caption",
                        "keywords",
                    }
                ):
                    try:
                        png_metadata.add_text(
                            metadata_key,
                            metadata_value,
                        )
                    except Exception:
                        pass

            png_metadata.add_text(
                "Title",
                title,
            )

            png_metadata.add_text(
                "Description",
                title,
            )

            png_metadata.add_text(
                "Keywords",
                keywords,
            )

            save_options: Dict[str, Any] = {
                "format": "PNG",
                "pnginfo": png_metadata,
                "compress_level": 6,
            }

            icc_profile = source_image.info.get(
                "icc_profile"
            )

            if icc_profile:
                save_options["icc_profile"] = icc_profile

            dpi = source_image.info.get("dpi")

            if dpi:
                save_options["dpi"] = dpi

            transparency = source_image.info.get(
                "transparency"
            )

            # ใช้ transparency option สำหรับ P/L/RGB
            # แต่ไม่ใช้กับ RGBA/LA เพราะ alpha อยู่ใน channel แล้ว
            if (
                transparency is not None
                and source_image.mode in (
                    "P",
                    "L",
                    "RGB",
                )
            ):
                save_options[
                    "transparency"
                ] = transparency

            output_buffer = io.BytesIO()

            # ไม่ convert mode และไม่ resize
            source_image.save(
                output_buffer,
                **save_options,
            )

            exported_bytes = output_buffer.getvalue()

        # ตรวจความถูกต้องหลัง export
        with Image.open(
            io.BytesIO(exported_bytes)
        ) as exported_image:
            if exported_image.size != original_size:
                raise ValueError(
                    "PNG dimensions changed during export"
                )

            # RGBA/LA ต้องยังคง alpha
            if original_mode in ("RGBA", "LA"):
                if exported_image.mode not in (
                    "RGBA",
                    "LA",
                ):
                    raise ValueError(
                        "PNG alpha channel was not preserved"
                    )

        return exported_bytes

    except Exception:
        # ความสำคัญสูงสุดคือไม่ทำลายไฟล์ต้นฉบับ
        return original_bytes


def export_image_with_metadata(
    original_bytes: bytes,
    original_extension: str,
    title: str,
    keywords: str,
) -> bytes:
    """
    เลือกวิธี Export ตามนามสกุลต้นฉบับ
    """
    extension = original_extension.lower()

    if extension in {".jpg", ".jpeg"}:
        return export_jpeg_with_iptc(
            original_bytes=original_bytes,
            title=title,
            keywords=keywords,
            original_extension=extension,
        )

    if extension == ".png":
        return export_png_with_metadata(
            original_bytes=original_bytes,
            title=title,
            keywords=keywords,
        )

    # ไม่ควรเกิดเพราะ uploader จำกัดชนิดไฟล์แล้ว
    return original_bytes


# =========================================================
# 14) SIDEBAR
# =========================================================
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "🔑 OpenAI API Key",
        value=os.getenv(
            "OPENAI_API_KEY",
            "",
        ),
        type="password",
        help=(
            "ใส่ API Key หรือกำหนด environment variable "
            "ชื่อ OPENAI_API_KEY"
        ),
    )

    selected_model_label = st.selectbox(
        "🤖 Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
    )

    selected_model_value = MODEL_OPTIONS[
        selected_model_label
    ]

    if selected_model_value == "custom":
        model_choice = st.text_input(
            "Model ID",
            value="gpt-5.6-terra",
            help=(
                "ตัวอย่าง: gpt-5.6, "
                "gpt-5.6-terra, gpt-5.6-luna"
            ),
        ).strip()
    else:
        model_choice = selected_model_value

    selected_category_name = st.selectbox(
        "📁 Adobe Category",
        options=list(CATEGORY_DICT.keys()),
        index=2,
    )

    selected_category_num = CATEGORY_DICT[
        selected_category_name
    ]

    st.divider()

    user_hint = st.text_area(
        "💡 Context Hint",
        placeholder=(
            "เช่น young woman stretching before tennis workout, "
            "healthy active lifestyle"
        ),
        height=100,
    )

    title_style = st.selectbox(
        "📝 Title Style",
        options=[
            "Natural commercial",
            "Clear descriptive",
            "Concise commercial",
            "Editorial safe",
        ],
        index=0,
    )

    keyword_style = st.selectbox(
        "🏷️ Keyword Strategy",
        options=[
            "Balanced",
            "Broad commercial reach",
            "Specific and accurate",
            "Concept focused",
        ],
        index=0,
    )

    blacklist_raw = st.text_area(
        "🛡️ Blacklist Keywords",
        value=DEFAULT_BLACKLIST,
        height=120,
    )

    blacklist_words = parse_blacklist(
        blacklist_raw
    )

    st.divider()

    st.markdown("**ติดตั้งแพ็กเกจ**")

    st.code(
        "pip install -U streamlit pandas pillow "
        "openai iptcinfo3"
    )

    st.caption(
        "แนะนำให้อัปเดต openai และ streamlit "
        "เป็นเวอร์ชันล่าสุดก่อนใช้งาน"
    )


# =========================================================
# 15) MAIN APPLICATION
# =========================================================
try:
    uploader_key = (
        f"image_uploader_"
        f"{st.session_state.uploader_version}"
    )

    uploaded_images = st.file_uploader(
        "📸 อัปโหลดรูปภาพ JPG, JPEG หรือ PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=uploader_key,
        help=(
            "ระบบจะรักษานามสกุล ขนาดภาพ "
            "และ transparency ของ PNG ตอน Export"
        ),
    )

    payloads: List[Dict[str, Any]] = []

    if uploaded_images:
        payloads = prepare_uploaded_payloads(
            uploaded_images
        )

        current_file_ids = {
            payload["id"]
            for payload in payloads
        }

        # ล้างเฉพาะผลของไฟล์ที่ผู้ใช้เอาออกจาก uploader
        for old_file_id in list(
            st.session_state.results.keys()
        ):
            if old_file_id not in current_file_ids:
                remove_result_widget_state(
                    old_file_id
                )

                del st.session_state.results[
                    old_file_id
                ]

        valid_payloads = [
            payload
            for payload in payloads
            if payload["valid"]
        ]

        invalid_payloads = [
            payload
            for payload in payloads
            if not payload["valid"]
        ]

        metric_col1, metric_col2, metric_col3 = (
            st.columns(3)
        )

        metric_col1.metric(
            "ไฟล์ทั้งหมด",
            len(payloads),
        )

        metric_col2.metric(
            "ไฟล์พร้อมวิเคราะห์",
            len(valid_payloads),
        )

        metric_col3.metric(
            "วิเคราะห์แล้ว",
            sum(
                1
                for payload in payloads
                if payload["id"]
                in st.session_state.results
            ),
        )

        if invalid_payloads:
            with st.expander(
                "⚠️ ไฟล์ที่เปิดไม่ได้"
            ):
                for payload in invalid_payloads:
                    st.error(
                        f"{payload['original_name']}: "
                        f"{payload['validation_error']}"
                    )

        action_col1, action_col2, action_col3 = (
            st.columns([1.2, 1, 1])
        )

        with action_col1:
            analyze_all_clicked = st.button(
                "🚀 วิเคราะห์ภาพทั้งหมด",
                type="primary",
                use_container_width=True,
            )

        with action_col2:
            clear_results_clicked = st.button(
                "♻️ ล้างเฉพาะผลวิเคราะห์",
                use_container_width=True,
            )

        with action_col3:
            clear_images_clicked = st.button(
                "🗑️ ลบภาพทั้งหมด",
                use_container_width=True,
            )

        if clear_results_clicked:
            for file_id in list(
                st.session_state.results.keys()
            ):
                remove_result_widget_state(file_id)

            st.session_state.results = {}
            st.session_state.generated_zip = None
            st.rerun()

        if clear_images_clicked:
            clear_all_uploaded_images()
            st.rerun()

        if analyze_all_clicked:
            if not api_key:
                st.error(
                    "กรุณาใส่ OpenAI API Key ก่อน"
                )

            elif not model_choice:
                st.error(
                    "กรุณาเลือกหรือกรอก Model ID"
                )

            elif not valid_payloads:
                st.error(
                    "ไม่มีไฟล์ภาพที่พร้อมวิเคราะห์"
                )

            else:
                progress_bar = st.progress(0)
                status_box = st.empty()

                total_valid_files = len(
                    valid_payloads
                )

                for index, payload in enumerate(
                    valid_payloads,
                    start=1,
                ):
                    status_box.info(
                        "กำลังวิเคราะห์ "
                        f"{index}/{total_valid_files}: "
                        f"{payload['original_name']}"
                    )

                    cache_key = (
                        make_analysis_cache_key(
                            image_bytes=payload[
                                "bytes"
                            ],
                            model=model_choice,
                            category_num=(
                                selected_category_num
                            ),
                            hint=user_hint,
                            blacklist_words=(
                                blacklist_words
                            ),
                            title_style=title_style,
                            keyword_style=(
                                keyword_style
                            ),
                        )
                    )

                    if (
                        cache_key
                        in st.session_state.analysis_cache
                    ):
                        analysis_result = (
                            st.session_state
                            .analysis_cache[
                                cache_key
                            ]
                        )
                    else:
                        analysis_result = (
                            analyze_image_with_openai(
                                image_bytes=payload[
                                    "bytes"
                                ],
                                category_name=(
                                    selected_category_name
                                ),
                                category_num=(
                                    selected_category_num
                                ),
                                hint=user_hint,
                                api_key=api_key,
                                model=model_choice,
                                blacklist_words=(
                                    blacklist_words
                                ),
                                title_style=(
                                    title_style
                                ),
                                keyword_style=(
                                    keyword_style
                                ),
                            )
                        )

                        st.session_state.analysis_cache[
                            cache_key
                        ] = analysis_result

                    save_analysis_result(
                        payload=payload,
                        result=analysis_result,
                        category_num=(
                            selected_category_num
                        ),
                    )

                    progress_bar.progress(
                        index / total_valid_files
                    )

                status_box.success(
                    "วิเคราะห์ภาพทั้งหมดเรียบร้อย"
                )

                st.session_state.generated_zip = None
                st.rerun()

    else:
        st.info(
            "อัปโหลดภาพเพื่อเริ่มสร้าง Title และ Keywords"
        )

    # =====================================================
    # 16) RESULT CARDS
    # =====================================================
    final_export_items: List[
        Dict[str, Any]
    ] = []

    if payloads:
        export_filename_map = (
            make_unique_export_filenames(payloads)
        )

        st.divider()
        st.subheader("📝 ผลลัพธ์")

        for payload in payloads:
            file_id = payload["id"]

            with st.container(border=True):
                image_col, content_col = st.columns(
                    [1, 2],
                    gap="large",
                )

                with image_col:
                    st.image(
                        payload["bytes"],
                        use_container_width=True,
                    )

                    st.caption(
                        payload["original_name"]
                    )

                    image_info = payload.get(
                        "image_info",
                        {},
                    )

                    if image_info:
                        width = image_info.get(
                            "width",
                            0,
                        )

                        height = image_info.get(
                            "height",
                            0,
                        )

                        mode = image_info.get(
                            "mode",
                            "-",
                        )

                        st.caption(
                            f"{width:,} × {height:,} px "
                            f"• {mode} "
                            f"• {payload['size'] / 1024:.1f} KB"
                        )

                        if image_info.get(
                            "has_transparency"
                        ):
                            st.success(
                                "PNG มีพื้นหลังโปร่งใส"
                            )

                    if not payload["valid"]:
                        st.error(
                            payload[
                                "validation_error"
                            ]
                            or "ไฟล์ไม่ถูกต้อง"
                        )

                with content_col:
                    if not payload["valid"]:
                        continue

                    existing_result = (
                        st.session_state.results.get(
                            file_id
                        )
                    )

                    button_col1, button_col2, button_col3 = (
                        st.columns(3)
                    )

                    with button_col1:
                        analyze_one_clicked = st.button(
                            "🔍 วิเคราะห์ภาพนี้",
                            key=(
                                f"analyze_one_"
                                f"{file_id}"
                            ),
                            use_container_width=True,
                        )

                    with button_col2:
                        regenerate_title_clicked = st.button(
                            "✨ สร้าง Title ใหม่",
                            key=(
                                f"regenerate_title_"
                                f"{file_id}"
                            ),
                            use_container_width=True,
                            disabled=(
                                existing_result is None
                            ),
                        )

                    with button_col3:
                        remove_result_clicked = st.button(
                            "🗑️ ลบผลภาพนี้",
                            key=(
                                f"remove_result_"
                                f"{file_id}"
                            ),
                            use_container_width=True,
                            disabled=(
                                existing_result is None
                            ),
                        )

                    # วิเคราะห์ภาพนี้ใหม่
                    if analyze_one_clicked:
                        if not api_key:
                            st.error(
                                "กรุณาใส่ OpenAI API Key ก่อน"
                            )
                        elif not model_choice:
                            st.error(
                                "กรุณาระบุ Model ID"
                            )
                        else:
                            with st.spinner(
                                "กำลังวิเคราะห์ภาพนี้..."
                            ):
                                cache_key = (
                                    make_analysis_cache_key(
                                        image_bytes=payload[
                                            "bytes"
                                        ],
                                        model=(
                                            model_choice
                                        ),
                                        category_num=(
                                            selected_category_num
                                        ),
                                        hint=user_hint,
                                        blacklist_words=(
                                            blacklist_words
                                        ),
                                        title_style=(
                                            title_style
                                        ),
                                        keyword_style=(
                                            keyword_style
                                        ),
                                    )
                                )

                                if (
                                    cache_key
                                    in st.session_state
                                    .analysis_cache
                                ):
                                    result = (
                                        st.session_state
                                        .analysis_cache[
                                            cache_key
                                        ]
                                    )
                                else:
                                    result = (
                                        analyze_image_with_openai(
                                            image_bytes=(
                                                payload[
                                                    "bytes"
                                                ]
                                            ),
                                            category_name=(
                                                selected_category_name
                                            ),
                                            category_num=(
                                                selected_category_num
                                            ),
                                            hint=user_hint,
                                            api_key=api_key,
                                            model=(
                                                model_choice
                                            ),
                                            blacklist_words=(
                                                blacklist_words
                                            ),
                                            title_style=(
                                                title_style
                                            ),
                                            keyword_style=(
                                                keyword_style
                                            ),
                                        )
                                    )

                                    st.session_state.analysis_cache[
                                        cache_key
                                    ] = result

                                save_analysis_result(
                                    payload=payload,
                                    result=result,
                                    category_num=(
                                        selected_category_num
                                    ),
                                )

                                st.session_state.generated_zip = (
                                    None
                                )

                            st.rerun()

                    # ลบผลเฉพาะภาพ
                    if remove_result_clicked:
                        st.session_state.results.pop(
                            file_id,
                            None,
                        )

                        remove_result_widget_state(
                            file_id
                        )

                        st.session_state.generated_zip = None
                        st.rerun()

                    # สร้าง Title ใหม่จาก Keywords ปัจจุบัน
                    if regenerate_title_clicked:
                        sync_widget_values_to_result(
                            file_id
                        )

                        current_result = (
                            st.session_state.results.get(
                                file_id
                            )
                        )

                        if not api_key:
                            st.error(
                                "กรุณาใส่ OpenAI API Key ก่อน"
                            )

                        elif not current_result:
                            st.error(
                                "ยังไม่มีผลวิเคราะห์"
                            )

                        else:
                            current_keywords = (
                                normalize_keywords(
                                    current_result.get(
                                        "keywords",
                                        "",
                                    ),
                                    blacklist_words=(
                                        blacklist_words
                                    ),
                                    limit=(
                                        KEYWORD_LIMIT
                                    ),
                                )
                            )

                            current_title = (
                                normalize_title(
                                    current_result.get(
                                        "title",
                                        "",
                                    )
                                )
                            )

                            if (
                                count_keywords(
                                    current_keywords
                                )
                                < TOP_KEYWORD_COUNT
                            ):
                                st.error(
                                    "ต้องมีอย่างน้อย "
                                    "10 Keywords "
                                    "ก่อนสร้าง Title ใหม่"
                                )
                            else:
                                with st.spinner(
                                    "กำลังสร้าง Title ใหม่ "
                                    "จาก 10 Keywords แรก..."
                                ):
                                    title_cache_key = (
                                        make_title_cache_key(
                                            keywords=(
                                                current_keywords
                                            ),
                                            current_title=(
                                                current_title
                                            ),
                                            model=(
                                                model_choice
                                            ),
                                            hint=user_hint,
                                            title_style=(
                                                title_style
                                            ),
                                            blacklist_words=(
                                                blacklist_words
                                            ),
                                        )
                                    )

                                    if (
                                        title_cache_key
                                        in st.session_state
                                        .title_cache
                                    ):
                                        title_result = (
                                            st.session_state
                                            .title_cache[
                                                title_cache_key
                                            ]
                                        )
                                    else:
                                        title_result = (
                                            regenerate_title_with_openai(
                                                keywords=(
                                                    current_keywords
                                                ),
                                                current_title=(
                                                    current_title
                                                ),
                                                hint=user_hint,
                                                api_key=(
                                                    api_key
                                                ),
                                                model=(
                                                    model_choice
                                                ),
                                                title_style=(
                                                    title_style
                                                ),
                                                blacklist_words=(
                                                    blacklist_words
                                                ),
                                            )
                                        )

                                        st.session_state.title_cache[
                                            title_cache_key
                                        ] = title_result

                                    if title_result[
                                        "error"
                                    ]:
                                        st.error(
                                            title_result[
                                                "error_message"
                                            ]
                                        )
                                    else:
                                        new_title = (
                                            title_result[
                                                "title"
                                            ]
                                        )

                                        st.session_state.results[
                                            file_id
                                        ][
                                            "title"
                                        ] = new_title

                                        st.session_state[
                                            f"title_{file_id}"
                                        ] = new_title

                                        st.session_state.results[
                                            file_id
                                        ][
                                            "keywords"
                                        ] = (
                                            current_keywords
                                        )

                                        st.session_state[
                                            f"keywords_{file_id}"
                                        ] = (
                                            current_keywords
                                        )

                                        st.session_state.generated_zip = (
                                            None
                                        )

                                        st.rerun()

                    existing_result = (
                        st.session_state.results.get(
                            file_id
                        )
                    )

                    if not existing_result:
                        st.info(
                            "ยังไม่มีผลวิเคราะห์สำหรับภาพนี้"
                        )
                        continue

                    if existing_result.get("error"):
                        st.error(
                            existing_result.get(
                                "error_message",
                                "เกิดข้อผิดพลาด",
                            )
                        )

                        with st.expander(
                            "ดู Error / Raw output"
                        ):
                            st.code(
                                existing_result.get(
                                    "raw",
                                    "",
                                )
                            )

                    title_widget_key = (
                        f"title_{file_id}"
                    )

                    keywords_widget_key = (
                        f"keywords_{file_id}"
                    )

                    if (
                        title_widget_key
                        not in st.session_state
                    ):
                        st.session_state[
                            title_widget_key
                        ] = existing_result.get(
                            "title",
                            "",
                        )

                    if (
                        keywords_widget_key
                        not in st.session_state
                    ):
                        st.session_state[
                            keywords_widget_key
                        ] = existing_result.get(
                            "keywords",
                            "",
                        )

                    edited_title = st.text_area(
                        "Title",
                        key=title_widget_key,
                        height=90,
                        help=(
                            "แก้ไขเองได้ หรือกด "
                            "“สร้าง Title ใหม่” "
                            "เพื่อใช้ 10 Keywords แรก"
                        ),
                    )

                    edited_keywords = st.text_area(
                        "Keywords",
                        key=keywords_widget_key,
                        height=150,
                        help=(
                            "เรียงคำสำคัญที่สุดไว้ 10 คำแรก "
                            "ก่อนกดสร้าง Title ใหม่"
                        ),
                    )

                    cleaned_title = normalize_title(
                        edited_title
                    )

                    cleaned_keywords = normalize_keywords(
                        edited_keywords,
                        blacklist_words=(
                            blacklist_words
                        ),
                        limit=KEYWORD_LIMIT,
                    )

                    # เก็บค่าที่แก้ล่าสุด
                    st.session_state.results[file_id][
                        "title"
                    ] = cleaned_title

                    st.session_state.results[file_id][
                        "keywords"
                    ] = cleaned_keywords

                    keyword_count = count_keywords(
                        cleaned_keywords
                    )

                    used_top_keywords = (
                        top_keywords_used_in_title(
                            title=cleaned_title,
                            keywords=cleaned_keywords,
                        )
                    )

                    title_errors = validate_title(
                        title=cleaned_title,
                        keywords=cleaned_keywords,
                        blacklist_words=(
                            blacklist_words
                        ),
                    )

                    keyword_errors = validate_keywords(
                        keywords=cleaned_keywords,
                        blacklist_words=(
                            blacklist_words
                        ),
                    )

                    quality_score = (
                        calculate_quality_score(
                            title=cleaned_title,
                            keywords=cleaned_keywords,
                            blacklist_words=(
                                blacklist_words
                            ),
                        )
                    )

                    result_metric1, result_metric2, result_metric3 = (
                        st.columns(3)
                    )

                    result_metric1.metric(
                        "Keywords",
                        f"{keyword_count}/{KEYWORD_LIMIT}",
                    )

                    result_metric2.metric(
                        "Top 10 ใน Title",
                        (
                            f"{len(used_top_keywords)}"
                            f"/{TOP_KEYWORD_COUNT}"
                        ),
                    )

                    result_metric3.metric(
                        "Quality Score",
                        quality_score,
                    )

                    with st.expander(
                        "🔎 ตรวจสอบ 10 Keywords แรก"
                    ):
                        top_keywords = (
                            get_keyword_list(
                                cleaned_keywords
                            )[
                                :TOP_KEYWORD_COUNT
                            ]
                        )

                        st.markdown(
                            "**10 Keywords แรก**"
                        )

                        st.write(
                            ", ".join(top_keywords)
                            if top_keywords
                            else "-"
                        )

                        st.markdown(
                            "**คำที่ตรวจพบใน Title**"
                        )

                        st.write(
                            ", ".join(
                                used_top_keywords
                            )
                            if used_top_keywords
                            else "-"
                        )

                    if not title_errors:
                        st.success(
                            "Title ผ่านเกณฑ์หลัก"
                        )
                    else:
                        for title_error in title_errors:
                            st.warning(
                                f"Title: {title_error}"
                            )

                    if not keyword_errors:
                        st.success(
                            "Keywords ครบและผ่านการตรวจพื้นฐาน"
                        )
                    else:
                        for keyword_error in (
                            keyword_errors
                        ):
                            st.warning(
                                "Keywords: "
                                f"{keyword_error}"
                            )

                    quality_notes = (
                        existing_result.get(
                            "quality_notes",
                            [],
                        )
                    )

                    if quality_notes:
                        with st.expander(
                            "Quality Notes"
                        ):
                            for note in quality_notes:
                                st.write(f"• {note}")

                    risk_notes = (
                        existing_result.get(
                            "risk_notes",
                            [],
                        )
                    )

                    if risk_notes:
                        with st.expander(
                            "Risk / Release Notes"
                        ):
                            for note in risk_notes:
                                st.write(f"• {note}")

                    final_export_items.append(
                        {
                            "file_id": file_id,
                            "filename": (
                                export_filename_map[
                                    file_id
                                ]
                            ),
                            "title": cleaned_title,
                            "keywords": (
                                cleaned_keywords
                            ),
                            "category": (
                                selected_category_num
                            ),
                            "releases": "",
                            "quality_score": (
                                quality_score
                            ),
                            "top_keyword_coverage": (
                                len(
                                    used_top_keywords
                                )
                            ),
                            "original_bytes": payload[
                                "bytes"
                            ],
                            "extension": payload[
                                "extension"
                            ],
                        }
                    )

    # =====================================================
    # 17) EXPORT
    # =====================================================
    if final_export_items:
        st.divider()
        st.subheader("📦 Export")

        # CSV มาตรฐาน Adobe Stock
        adobe_export_df = pd.DataFrame(
            [
                {
                    "Filename": item[
                        "filename"
                    ],
                    "Title": item["title"],
                    "Keywords": item[
                        "keywords"
                    ],
                    "Category": item[
                        "category"
                    ],
                    "Releases": item[
                        "releases"
                    ],
                }
                for item in final_export_items
            ]
        )

        # ตารางตรวจสอบภายใน ไม่ใส่ใน Adobe CSV หลัก
        review_df = pd.DataFrame(
            [
                {
                    "Filename": item[
                        "filename"
                    ],
                    "Title Length": len(
                        item["title"]
                    ),
                    "Keyword Count": (
                        count_keywords(
                            item["keywords"]
                        )
                    ),
                    "Top Keyword Coverage": (
                        item[
                            "top_keyword_coverage"
                        ]
                    ),
                    "Quality Score": item[
                        "quality_score"
                    ],
                    "Original Format": (
                        item[
                            "extension"
                        ].replace(".", "").upper()
                    ),
                }
                for item in final_export_items
            ]
        )

        csv_bytes = adobe_export_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            st.download_button(
                "📊 Download Adobe CSV",
                data=csv_bytes,
                file_name=(
                    "adobe_stock_metadata.csv"
                ),
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

            progress_bar = st.progress(0)
            export_status = st.empty()

            with zipfile.ZipFile(
                zip_buffer,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as zip_file:
                zip_file.writestr(
                    "adobe_stock_metadata.csv",
                    csv_bytes,
                )

                review_csv_bytes = (
                    review_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")
                )

                zip_file.writestr(
                    "metadata_quality_review.csv",
                    review_csv_bytes,
                )

                total_items = len(
                    final_export_items
                )

                for index, item in enumerate(
                    final_export_items,
                    start=1,
                ):
                    export_status.info(
                        "กำลังเตรียมไฟล์ "
                        f"{index}/{total_items}: "
                        f"{item['filename']}"
                    )

                    exported_image_bytes = (
                        export_image_with_metadata(
                            original_bytes=item[
                                "original_bytes"
                            ],
                            original_extension=item[
                                "extension"
                            ],
                            title=item["title"],
                            keywords=item[
                                "keywords"
                            ],
                        )
                    )

                    zip_file.writestr(
                        item["filename"],
                        exported_image_bytes,
                    )

                    progress_bar.progress(
                        index / total_items
                    )

            zip_buffer.seek(0)

            st.session_state.generated_zip = (
                zip_buffer.getvalue()
            )

            st.session_state.generated_zip_name = (
                "adobe_stock_package.zip"
            )

            export_status.success(
                "สร้าง ZIP เรียบร้อย"
            )

        if st.session_state.generated_zip:
            st.download_button(
                "📂 Download ZIP",
                data=(
                    st.session_state.generated_zip
                ),
                file_name=(
                    st.session_state
                    .generated_zip_name
                    or "adobe_stock_package.zip"
                ),
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

        with st.expander(
            "ดู Adobe CSV ก่อนดาวน์โหลด"
        ):
            st.dataframe(
                adobe_export_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "ดู Quality Review"
        ):
            st.dataframe(
                review_df,
                use_container_width=True,
                hide_index=True,
            )


except Exception:
    st.error("Application Error")

    with st.expander(
        "ดูรายละเอียด Error",
        expanded=True,
    ):
        st.code(traceback.format_exc())