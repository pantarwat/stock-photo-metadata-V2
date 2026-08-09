from __future__ import annotations

"""
AI Stock Vision V2.5 — Cloud-safe single-file Streamlit application

Run:
    pip install -r requirements.txt
    streamlit run app.py

Security:
    Never hard-code or commit OPENAI_API_KEY.
"""

# ========================================================================================
# Inlined from: core/config.py
# ========================================================================================

import tempfile
from pathlib import Path

APP_NAME = "AI Stock Vision V2.5 — Cloud Safe"
PROMPT_VERSION = "v2.5.0-cloud-safe"
ANALYSIS_MAX_LONG_EDGE = 1800
TOKEN_LADDER = (2400, 3600, 5000)
TITLE_REPAIR_MAX_ATTEMPTS = 3

# Model selection
# AUTO_MODEL_OPTION is a UI/internal sentinel and is never sent to the API.
AUTO_MODEL_OPTION = "__auto_best_compatible__"
DEFAULT_MODEL_ID = "gpt-5.6"
LEGACY_VISION_FALLBACK_MODEL_ID = "gpt-4.1"
DEFAULT_AI_REVIEW_THRESHOLD = 75
DEFAULT_USD_TO_THB = 35.0

# Ordered from highest expected metadata quality to lower-cost / older fallbacks.
# A candidate is only preferred automatically when the API key can see it.
MODEL_QUALITY_PRIORITY = (
    # Current official high-quality vision-capable families, ordered for this task.
    "gpt-5.6",          # alias of GPT-5.6 Sol
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-4.1",
    "gpt-4o",
    # Lower-cost fallbacks after the quality-oriented choices above.
    "gpt-5.6-luna",
    "gpt-5.4-mini",
    "gpt-4.1-mini",
    "gpt-4o-mini",
    "gpt-5.4-nano",
)

# Used only when model discovery cannot run. Each candidate is attempted safely;
# model-not-found / unsupported-image errors automatically move to the next one.
MODEL_DISCOVERY_FAILURE_CHAIN = (
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5.6-luna",
    "gpt-5.4-mini",
    "gpt-4.1-mini",
    "gpt-4o-mini",
)

APP_CACHE_DIR = Path(tempfile.gettempdir()) / "ai_stock_vision_cache"
ANALYSIS_CACHE_DIR = APP_CACHE_DIR / "analysis"
TITLE_CACHE_DIR = APP_CACHE_DIR / "title"
QUALITY_CACHE_DIR = APP_CACHE_DIR / "quality"
EXPORT_CACHE_DIR = APP_CACHE_DIR / "exports"

CACHE_DIRS = (
    ANALYSIS_CACHE_DIR,
    TITLE_CACHE_DIR,
    QUALITY_CACHE_DIR,
    EXPORT_CACHE_DIR,
)

DEFAULT_BLACKLIST = {
    "adobe stock",
    "shutterstock",
    "getty images",
    "istock",
    "alamy",
    "dreamstime",
    "depositphotos",
    "copyright",
    "copyrighted",
    "trademark",
    "watermark",
    "logo",
    "brand",
    "branded",
    "celebrity",
    "famous person",
    "ai generated",
    "generative ai",
    "midjourney",
    "dall-e",
    "stable diffusion",
}

OBVIOUS_NON_VISION_MODEL_TOKENS = (
    "embedding",
    "whisper",
    "tts",
    "audio",
    "transcribe",
    "moderation",
    "realtime",
    "search-preview",
    "image-1",
    "sora",
    "codex",
    "-pro",
    "chat",
)

# ========================================================================================
# Inlined from: core/cache.py
# ========================================================================================

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any



def ensure_cache_dirs() -> None:
    for directory in CACHE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def make_cache_key(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(directory: Path, *parts: object) -> Path:
    ensure_cache_dirs()
    key = make_cache_key(PROMPT_VERSION, *parts)
    return directory / f"{key}.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temp_path.replace(path)
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def directory_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    if not path.exists():
        return file_count, total_bytes
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            try:
                total_bytes += item.stat().st_size
            except OSError:
                pass
    return file_count, total_bytes


def clear_app_cache() -> tuple[int, int]:
    """Delete only this application's cache directory and recreate safe subfolders."""
    expected = (Path(tempfile.gettempdir()) / "ai_stock_vision_cache").resolve()
    actual = APP_CACHE_DIR.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()

    if actual != expected or actual == temp_root or temp_root not in actual.parents:
        raise RuntimeError("ปฏิเสธการล้างแคช: ตำแหน่งโฟลเดอร์ไม่ผ่านการตรวจสอบความปลอดภัย")

    file_count, total_bytes = directory_stats(APP_CACHE_DIR)
    if APP_CACHE_DIR.exists():
        shutil.rmtree(APP_CACHE_DIR)
    ensure_cache_dirs()
    return file_count, total_bytes


def human_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"

# ========================================================================================
# Inlined from: core/image_utils.py
# ========================================================================================

import base64
import hashlib
import io
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError



def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def has_transparency(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        return True
    if image.mode == "P" and "transparency" in image.info:
        return True
    return False


def inspect_image(data: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return {
                "width": int(image.width),
                "height": int(image.height),
                "mode": str(image.mode),
                "format": str(image.format or ""),
                "transparency": has_transparency(image),
                "dpi": image.info.get("dpi"),
                "has_icc_profile": bool(image.info.get("icc_profile")),
            }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("ไม่สามารถอ่านไฟล์ภาพนี้ได้ หรือไฟล์อาจเสียหาย") from exc


def create_analysis_data_url(
    original_bytes: bytes,
    max_long_edge: int = ANALYSIS_MAX_LONG_EDGE,
) -> tuple[str, dict[str, Any]]:
    """Create a resized analysis copy. The returned bytes are never used for export."""
    try:
        with Image.open(io.BytesIO(original_bytes)) as source:
            source.load()
            source = ImageOps.exif_transpose(source)
            alpha = has_transparency(source)

            if alpha:
                working = source.convert("RGBA")
                output_format = "PNG"
                media_type = "image/png"
            else:
                working = source.convert("RGB")
                output_format = "JPEG"
                media_type = "image/jpeg"

            original_size = working.size
            working.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            if output_format == "PNG":
                working.save(output, format="PNG", optimize=True)
            else:
                working.save(
                    output,
                    format="JPEG",
                    quality=88,
                    optimize=True,
                    progressive=True,
                )

            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            data_url = f"data:{media_type};base64,{encoded}"
            info = {
                "analysis_width": working.width,
                "analysis_height": working.height,
                "analysis_format": output_format,
                "original_width": original_size[0],
                "original_height": original_size[1],
            }
            return data_url, info
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("สร้างสำเนาภาพสำหรับวิเคราะห์ไม่สำเร็จ") from exc

# ========================================================================================
# Inlined from: core/quality.py
# ========================================================================================

import re
from collections import Counter
from typing import Any, Iterable

FORBIDDEN_TITLE_PREFIXES = ("image of", "photo of", "picture of")
CONNECTOR_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "with",
    "in",
    "on",
    "at",
    "by",
    "for",
    "from",
    "to",
    "of",
    "during",
    "before",
    "after",
    "while",
    "under",
    "over",
    "near",
    "through",
    "into",
    "using",
    "showing",
    "featuring",
}


def normalize_phrase(value: str) -> str:
    value = value.casefold().strip()
    value = value.replace("&", " and ")
    value = re.sub(r"[-_/]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def keyword_in_title(keyword: str, title: str) -> bool:
    keyword_norm = normalize_phrase(keyword)
    title_norm = normalize_phrase(title)
    if not keyword_norm or not title_norm:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(keyword_norm)}(?![a-z0-9])"
    return re.search(pattern, title_norm) is not None


def parse_keywords(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = re.split(r"[,;\n|]+", value)
    else:
        raw_items = [str(item) for item in value]

    cleaned: list[str] = []
    for item in raw_items:
        item = re.sub(r"^\s*\d+[\.)-]?\s*", "", item)
        item = item.strip().strip('"\'`[]{}()')
        item = re.sub(r"\s+", " ", item)
        if item:
            cleaned.append(item)
    return cleaned


def find_duplicates(keywords: list[str]) -> list[str]:
    normalized = [normalize_phrase(keyword) for keyword in keywords]
    counts = Counter(item for item in normalized if item)
    duplicates: list[str] = []
    seen: set[str] = set()
    for original, normalized_value in zip(keywords, normalized):
        if counts[normalized_value] > 1 and normalized_value not in seen:
            duplicates.append(original)
            seen.add(normalized_value)
    return duplicates


def find_blacklist_hits(keywords: list[str], blacklist: Iterable[str]) -> list[str]:
    blacklist_norm = {normalize_phrase(item) for item in blacklist if normalize_phrase(item)}
    hits: list[str] = []
    for keyword in keywords:
        normalized = normalize_phrase(keyword)
        if normalized in blacklist_norm:
            hits.append(keyword)
    return hits


def top_keyword_coverage(title: str, keywords: list[str], top_n: int = 10) -> dict[str, Any]:
    top = keywords[:top_n]
    present = [keyword for keyword in top if keyword_in_title(keyword, title)]
    missing = [keyword for keyword in top if not keyword_in_title(keyword, title)]
    return {
        "count": len(present),
        "total": len(top),
        "present": present,
        "missing": missing,
    }


def looks_like_keyword_dump(title: str) -> bool:
    words = normalize_phrase(title).split()
    if len(words) < 8:
        return False
    connector_count = sum(word in CONNECTOR_WORDS for word in words)
    comma_count = title.count(",")
    return comma_count >= 6 or (len(words) >= 14 and connector_count <= 2)


def ai_average(ai_review: dict[str, Any] | None) -> float | None:
    if not ai_review:
        return None
    values: list[float] = []
    for key in ("relevance_score", "naturalness_score", "keyword_accuracy_score"):
        try:
            values.append(float(ai_review[key]))
        except (KeyError, TypeError, ValueError):
            return None
    return sum(values) / len(values)


def evaluate_quality(
    title: str,
    keywords: list[str],
    blacklist: Iterable[str],
    ai_review: dict[str, Any] | None = None,
    ai_threshold: int = 75,
) -> dict[str, Any]:
    title = (title or "").strip()
    keywords = parse_keywords(keywords)
    duplicate_keywords = find_duplicates(keywords)
    blacklist_hits = find_blacklist_hits(keywords, blacklist)
    coverage = top_keyword_coverage(title, keywords, top_n=10)
    title_length = len(title)
    comma_count = title.count(",")
    prefix_bad = normalize_phrase(title).startswith(FORBIDDEN_TITLE_PREFIXES)
    keyword_dump = looks_like_keyword_dump(title)

    issues: list[str] = []
    warnings: list[str] = []

    if len(keywords) != 49:
        issues.append(f"Keywords มี {len(keywords)} คำ ต้องเป็น 49 คำ")
    if duplicate_keywords:
        issues.append("พบ Keyword ซ้ำ: " + ", ".join(duplicate_keywords))
    if blacklist_hits:
        issues.append("พบคำ Blacklist: " + ", ".join(blacklist_hits))
    if not title:
        issues.append("Title ว่าง")
    if title_length > 200:
        issues.append(f"Title ยาว {title_length} ตัวอักษร เกิน 200")
    if coverage["count"] != 10 or coverage["total"] != 10:
        missing = ", ".join(coverage["missing"]) or "Top 10 ยังไม่ครบ"
        issues.append(f"Top 10 Keyword Coverage ไม่ครบ: {missing}")
    if prefix_bad:
        issues.append("Title ขึ้นต้นด้วย Image of / Photo of / Picture of")
    if comma_count > 5:
        warnings.append("Title มี comma มากเกินไป")
    if keyword_dump:
        warnings.append("Title มีลักษณะคล้าย Keyword Dump")

    keyword_score = 20 if len(keywords) == 49 else round(20 * min(len(keywords), 49) / 49, 2)
    coverage_score = round(35 * min(coverage["count"], 10) / 10, 2)
    title_score = 10 if title and title_length <= 200 else 0
    duplicate_score = 10 if not duplicate_keywords else 0
    blacklist_score = 10 if not blacklist_hits else 0

    average = ai_average(ai_review)
    ai_score = round(15 * average / 100, 2) if average is not None else 0
    score = round(
        keyword_score
        + coverage_score
        + title_score
        + duplicate_score
        + blacklist_score
        + ai_score,
        2,
    )

    hard_pass = (
        len(keywords) == 49
        and coverage["count"] == 10
        and coverage["total"] == 10
        and bool(title)
        and title_length <= 200
        and not duplicate_keywords
        and not blacklist_hits
        and not prefix_bad
    )
    ai_pass = average is None or average >= ai_threshold
    export_eligible = hard_pass and ai_pass

    if not hard_pass or not ai_pass:
        status = "ไม่ผ่าน — ควรแก้ก่อน Export"
    elif average is None:
        status = "ควรตรวจสอบเพิ่มเติม — ยังไม่ได้ AI Review"
    elif score >= 90:
        status = "ผ่าน — พร้อม Export"
    else:
        status = "ควรตรวจสอบเพิ่มเติม"

    if average is not None and average < ai_threshold:
        issues.append(f"AI Review เฉลี่ย {average:.0f} ต่ำกว่าเกณฑ์ {ai_threshold}")

    return {
        "score": score,
        "status": status,
        "hard_pass": hard_pass,
        "ai_pass": ai_pass,
        "export_eligible": export_eligible,
        "keyword_count": len(keywords),
        "title_length": title_length,
        "top10_coverage": coverage["count"],
        "top10_total": coverage["total"],
        "missing_keywords": coverage["missing"],
        "duplicate_keywords": duplicate_keywords,
        "blacklist_hits": blacklist_hits,
        "comma_count": comma_count,
        "keyword_dump": keyword_dump,
        "ai_average": average,
        "issues": issues,
        "warnings": warnings,
        "score_breakdown": {
            "keyword_count": keyword_score,
            "top10_coverage": coverage_score,
            "title_length": title_score,
            "duplicates": duplicate_score,
            "blacklist": blacklist_score,
            "ai_review": ai_score,
        },
    }

# ========================================================================================
# Inlined from: core/costs.py
# ========================================================================================

from collections import defaultdict
from typing import Any, Iterable

# Prices are USD per 1 million tokens. Keep this table explicit and reviewable.
# Updated from official OpenAI model pages on 2026-07-11.
MODEL_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # USD per 1M tokens. Verify periodically against official OpenAI pricing.
    "gpt-5.6": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}
PRICING_UPDATED_AT = "2026-08-09"


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def resolve_pricing_key(model_id: str) -> str | None:
    """Resolve stable aliases and dated snapshots without guessing unknown families."""
    model = str(model_id or "").strip().casefold()
    if not model:
        return None
    for key in sorted(MODEL_PRICING_USD_PER_MTOK, key=len, reverse=True):
        if model == key or model.startswith(key + "-"):
            return key
    return None


def extract_response_usage(response: Any) -> dict[str, int]:
    """Read Responses API usage across SDK objects and plain dictionaries."""
    usage = get_attr(response, "usage", None) or {}
    input_details = (
        get_attr(usage, "input_tokens_details", None)
        or get_attr(usage, "prompt_tokens_details", None)
        or {}
    )
    output_details = get_attr(usage, "output_tokens_details", None) or {}

    input_tokens = safe_int(
        get_attr(usage, "input_tokens", None)
        or get_attr(usage, "prompt_tokens", None)
    )
    output_tokens = safe_int(
        get_attr(usage, "output_tokens", None)
        or get_attr(usage, "completion_tokens", None)
    )
    cached_input_tokens = min(
        input_tokens,
        safe_int(get_attr(input_details, "cached_tokens", 0)),
    )
    reasoning_tokens = safe_int(get_attr(output_details, "reasoning_tokens", 0))
    total_tokens = safe_int(get_attr(usage, "total_tokens", 0))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "non_cached_input_tokens": max(0, input_tokens - cached_input_tokens),
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def calculate_usage_cost(model_id: str, usage: dict[str, int]) -> dict[str, Any]:
    pricing_key = resolve_pricing_key(model_id)
    if pricing_key is None:
        return {
            "pricing_available": False,
            "pricing_key": None,
            "pricing_updated_at": PRICING_UPDATED_AT,
            "input_cost_usd": None,
            "cached_input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
        }

    price = MODEL_PRICING_USD_PER_MTOK[pricing_key]
    input_cost = usage["non_cached_input_tokens"] * price["input"] / 1_000_000
    cached_cost = usage["cached_input_tokens"] * price["cached_input"] / 1_000_000
    output_cost = usage["output_tokens"] * price["output"] / 1_000_000
    return {
        "pricing_available": True,
        "pricing_key": pricing_key,
        "pricing_updated_at": PRICING_UPDATED_AT,
        "price_per_million": dict(price),
        "input_cost_usd": input_cost,
        "cached_input_cost_usd": cached_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + cached_cost + output_cost,
    }


def build_usage_event(
    response: Any,
    *,
    model: str,
    operation: str,
    method: str,
    token_limit: int,
) -> dict[str, Any]:
    usage = extract_response_usage(response)
    cost = calculate_usage_cost(model, usage)
    response_id = str(get_attr(response, "id", "") or "").strip()
    return {
        "event_id": response_id,
        "operation": operation,
        "method": method,
        "model": model,
        "token_limit": token_limit,
        "usage": usage,
        "cost": cost,
        "api_response_received": True,
    }


def usage_events_from_attempts(attempts: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for attempt in attempts or []:
        if not isinstance(attempt, dict):
            continue
        event = attempt.get("usage_event")
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "")
        # Response IDs are ideal for de-duplication. Fall back to object position when absent.
        dedupe_key = event_id or f"anonymous-{id(event)}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        events.append(dict(event))
    return events


def merge_usage_events(existing: Iterable[dict[str, Any]], new: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in [*(existing or []), *(new or [])]:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "")
        if event_id:
            key = event_id
        else:
            usage = event.get("usage") or {}
            key = "|".join(
                [
                    str(event.get("operation") or ""),
                    str(event.get("method") or ""),
                    str(event.get("model") or ""),
                    str(usage.get("input_tokens") or 0),
                    str(usage.get("output_tokens") or 0),
                    str(len(output)),
                ]
            )
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(event))
    return output


def summarize_usage_events(events: Iterable[dict[str, Any]], usd_to_thb: float = 0.0) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "api_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "non_cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "known_cost_usd": 0.0,
        "estimated_cost_thb": 0.0,
        "unknown_pricing_models": [],
        "fully_priced": True,
        "by_operation": {},
        "by_model": {},
    }
    unknown: set[str] = set()
    operation_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "known_cost_usd": 0.0}
    )
    model_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "known_cost_usd": 0.0}
    )

    for event in events or []:
        if not isinstance(event, dict) or not event.get("api_response_received"):
            continue
        usage = event.get("usage") or {}
        cost = event.get("cost") or {}
        model = str(event.get("model") or "unknown")
        operation = str(event.get("operation") or "api_call")
        summary["api_calls"] += 1
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "non_cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            summary[key] += safe_int(usage.get(key, 0))

        usd = cost.get("total_cost_usd")
        if usd is None:
            unknown.add(model)
        else:
            summary["known_cost_usd"] += float(usd)

        for bucket in (operation_buckets[operation], model_buckets[model]):
            bucket["api_calls"] += 1
            bucket["input_tokens"] += safe_int(usage.get("input_tokens", 0))
            bucket["output_tokens"] += safe_int(usage.get("output_tokens", 0))
            if usd is not None:
                bucket["known_cost_usd"] += float(usd)

    summary["unknown_pricing_models"] = sorted(unknown)
    summary["fully_priced"] = not bool(unknown)
    summary["estimated_cost_thb"] = summary["known_cost_usd"] * max(0.0, float(usd_to_thb or 0.0))
    summary["by_operation"] = dict(operation_buckets)
    summary["by_model"] = dict(model_buckets)
    return summary


def empty_usage_summary(usd_to_thb: float = 0.0) -> dict[str, Any]:
    return summarize_usage_events([], usd_to_thb)

# ========================================================================================
# Inlined from: core/exporter.py
# ========================================================================================

import binascii
import csv
import io
import json
import re
import shutil
import struct
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape



def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[\x00-\x1f]", "_", name)
    return name or "image"


def unique_filename(filename: str, used: set[str]) -> str:
    filename = safe_filename(filename)
    path = Path(filename)
    candidate = filename
    index = 2
    while candidate.casefold() in used:
        candidate = f"{path.stem}_{index}{path.suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _png_itxt(keyword: str, text: str) -> bytes:
    keyword_bytes = keyword.encode("latin-1", errors="replace")[:79]
    text_bytes = text.encode("utf-8")
    data = (
        keyword_bytes
        + b"\x00"
        + b"\x00"  # uncompressed
        + b"\x00"  # compression method
        + b"\x00"  # empty language tag
        + b"\x00"  # empty translated keyword
        + text_bytes
    )
    return _png_chunk(b"iTXt", data)


def _xmp_packet(title: str, keywords: list[str]) -> str:
    subject_items = "".join(f"<rdf:li>{escape(keyword)}</rdf:li>" for keyword in keywords)
    return (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{escape(title)}</rdf:li></rdf:Alt></dc:title>'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{escape(title)}</rdf:li></rdf:Alt></dc:description>'
        f'<dc:subject><rdf:Bag>{subject_items}</rdf:Bag></dc:subject>'
        '</rdf:Description></rdf:RDF></x:xmpmeta>'
        '<?xpacket end="w"?>'
    )


def embed_png_metadata(original_bytes: bytes, title: str, keywords: list[str]) -> bytes:
    """Insert iTXt chunks without decoding or re-encoding PNG pixel data."""
    signature = b"\x89PNG\r\n\x1a\n"
    if not original_bytes.startswith(signature):
        raise ValueError("ไฟล์ PNG ไม่ถูกต้อง")

    output = bytearray(signature)
    offset = len(signature)
    inserted = False
    replacement_keys = {"title", "description", "keywords", "xml:com.adobe.xmp"}

    while offset + 12 <= len(original_bytes):
        length = struct.unpack(">I", original_bytes[offset : offset + 4])[0]
        chunk_end = offset + 12 + length
        if chunk_end > len(original_bytes):
            raise ValueError("โครงสร้าง PNG ไม่สมบูรณ์")

        chunk_type = original_bytes[offset + 4 : offset + 8]
        chunk_data = original_bytes[offset + 8 : offset + 8 + length]
        keep = True

        if chunk_type in {b"tEXt", b"zTXt", b"iTXt"}:
            raw_key = chunk_data.split(b"\x00", 1)[0]
            key = raw_key.decode("latin-1", errors="ignore").casefold()
            if key in replacement_keys:
                keep = False

        if chunk_type == b"IEND" and not inserted:
            output.extend(_png_itxt("Title", title))
            output.extend(_png_itxt("Description", title))
            output.extend(_png_itxt("Keywords", ", ".join(keywords)))
            output.extend(_png_itxt("XML:com.adobe.xmp", _xmp_packet(title, keywords)))
            inserted = True

        if keep:
            output.extend(original_bytes[offset:chunk_end])
        offset = chunk_end

        if chunk_type == b"IEND":
            break

    if not inserted:
        raise ValueError("ไม่พบ IEND chunk ในไฟล์ PNG")
    return bytes(output)


def embed_jpeg_iptc(original_bytes: bytes, suffix: str, title: str, keywords: list[str]) -> bytes:
    try:
        from iptcinfo3 import IPTCInfo
    except ImportError as exc:
        raise RuntimeError("ยังไม่ได้ติดตั้ง IPTCInfo3 จึงไม่สามารถฝัง IPTC ใน JPEG") from exc

    with tempfile.TemporaryDirectory(prefix="ai_stock_vision_iptc_") as temp_dir:
        input_path = Path(temp_dir) / f"input{suffix.lower()}"
        output_path = Path(temp_dir) / f"output{suffix.lower()}"
        input_path.write_bytes(original_bytes)
        shutil.copy2(input_path, output_path)

        info = IPTCInfo(str(output_path), force=True, inp_charset="utf8", out_charset="utf8")
        info["object name"] = title
        info["caption/abstract"] = title
        info["keywords"] = list(keywords)
        info.save_as(str(output_path), {"overwrite": True})

        backup = Path(f"{output_path}~")
        if backup.exists():
            backup.unlink()
        return output_path.read_bytes()


def embed_metadata(record: dict[str, Any]) -> tuple[bytes, str | None]:
    original_bytes = record["bytes"]
    title = str(record.get("title") or "").strip()
    keywords = parse_keywords(record.get("keywords"))
    suffix = Path(record["filename"]).suffix.casefold()

    try:
        if suffix == ".png":
            return embed_png_metadata(original_bytes, title, keywords), None
        if suffix in {".jpg", ".jpeg"}:
            return embed_jpeg_iptc(original_bytes, suffix, title, keywords), None
        return original_bytes, "ไม่รองรับการฝัง Metadata สำหรับนามสกุลนี้ จึงใช้ไฟล์ต้นฉบับ"
    except Exception as exc:
        return original_bytes, f"ฝัง Metadata ไม่สำเร็จ จึงใช้ไฟล์ต้นฉบับ: {type(exc).__name__}: {exc}"


def adobe_csv_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=["Filename", "Title", "Keywords", "Category", "Releases"],
    )
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                "Filename": record["export_filename"],
                "Title": str(record.get("title") or "").strip(),
                "Keywords": ", ".join(parse_keywords(record.get("keywords"))),
                "Category": str(record.get("category") or "").strip(),
                "Releases": str(record.get("releases") or "").strip(),
            }
        )
    return output.getvalue().encode("utf-8-sig")


def quality_csv_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "Filename",
        "Quality Score",
        "Quality Status",
        "Keyword Count",
        "Top 10 Coverage",
        "Title Length",
        "AI Relevance Score",
        "AI Naturalness Score",
        "AI Keyword Accuracy Score",
        "AI Average",
        "API Calls",
        "Input Tokens",
        "Cached Input Tokens",
        "Output Tokens",
        "API Cost USD",
        "Estimated Cost THB",
        "Unknown Pricing Models",
        "Issues",
        "Warnings",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        quality = record.get("quality") or {}
        review = record.get("ai_review") or {}
        usage_summary = record.get("api_usage_summary") or summarize_usage_events(
            record.get("api_usage_events", [])
        )
        writer.writerow(
            {
                "Filename": record["export_filename"],
                "Quality Score": quality.get("score", ""),
                "Quality Status": quality.get("status", ""),
                "Keyword Count": quality.get("keyword_count", ""),
                "Top 10 Coverage": f"{quality.get('top10_coverage', 0)}/{quality.get('top10_total', 10)}",
                "Title Length": quality.get("title_length", ""),
                "AI Relevance Score": review.get("relevance_score", ""),
                "AI Naturalness Score": review.get("naturalness_score", ""),
                "AI Keyword Accuracy Score": review.get("keyword_accuracy_score", ""),
                "AI Average": ai_average(review) if review else "",
                "API Calls": usage_summary.get("api_calls", 0),
                "Input Tokens": usage_summary.get("input_tokens", 0),
                "Cached Input Tokens": usage_summary.get("cached_input_tokens", 0),
                "Output Tokens": usage_summary.get("output_tokens", 0),
                "API Cost USD": f"{float(usage_summary.get('known_cost_usd', 0.0)):.8f}",
                "Estimated Cost THB": f"{float(usage_summary.get('estimated_cost_thb', 0.0)):.6f}",
                "Unknown Pricing Models": ", ".join(usage_summary.get("unknown_pricing_models", [])),
                "Issues": " | ".join(quality.get("issues", [])),
                "Warnings": " | ".join(quality.get("warnings", [])),
            }
        )
    return output.getvalue().encode("utf-8-sig")


def api_cost_csv_bytes(records: Iterable[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "Filename",
        "Operation",
        "Model",
        "Method",
        "Input Tokens",
        "Cached Input Tokens",
        "Non-cached Input Tokens",
        "Output Tokens",
        "Reasoning Tokens",
        "Total Tokens",
        "Pricing Available",
        "Pricing Key",
        "Cost USD",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        for event in record.get("api_usage_events", []) or []:
            usage = event.get("usage") or {}
            cost = event.get("cost") or {}
            writer.writerow(
                {
                    "Filename": record["export_filename"],
                    "Operation": event.get("operation", ""),
                    "Model": event.get("model", ""),
                    "Method": event.get("method", ""),
                    "Input Tokens": usage.get("input_tokens", 0),
                    "Cached Input Tokens": usage.get("cached_input_tokens", 0),
                    "Non-cached Input Tokens": usage.get("non_cached_input_tokens", 0),
                    "Output Tokens": usage.get("output_tokens", 0),
                    "Reasoning Tokens": usage.get("reasoning_tokens", 0),
                    "Total Tokens": usage.get("total_tokens", 0),
                    "Pricing Available": cost.get("pricing_available", False),
                    "Pricing Key": cost.get("pricing_key") or "",
                    "Cost USD": (
                        f"{float(cost['total_cost_usd']):.8f}"
                        if cost.get("total_cost_usd") is not None
                        else ""
                    ),
                }
            )
    return output.getvalue().encode("utf-8-sig")


def build_export_zip(records: list[dict[str, Any]]) -> dict[str, Any]:
    EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_path = EXPORT_CACHE_DIR / f"ai_stock_vision_export_{timestamp}.zip"
    used_names: set[str] = set()
    export_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for record in records:
        export_record = dict(record)
        export_record["export_filename"] = unique_filename(record["filename"], used_names)
        export_records.append(export_record)

    adobe_csv = adobe_csv_bytes(export_records)
    quality_csv = quality_csv_bytes(export_records)
    api_cost_csv = api_cost_csv_bytes(export_records)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for record in export_records:
            embedded_bytes, warning = embed_metadata(record)
            archive.writestr(f"images/{record['export_filename']}", embedded_bytes)
            if warning:
                warnings.append(f"{record['export_filename']}: {warning}")
        archive.writestr("adobe_stock_metadata.csv", adobe_csv)
        archive.writestr("quality_report.csv", quality_csv)
        archive.writestr("api_cost_report.csv", api_cost_csv)
        if warnings:
            archive.writestr(
                "export_warnings.txt",
                "\n".join(warnings).encode("utf-8"),
            )

    return {
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.read_bytes(),
        "filename": zip_path.name,
        "adobe_csv": adobe_csv,
        "quality_csv": quality_csv,
        "api_cost_csv": api_cost_csv,
        "warnings": warnings,
        "exported_count": len(export_records),
    }

# ========================================================================================
# Inlined from: core/openai_service.py
# ========================================================================================

import json
import re
from typing import Any, Callable, Iterable, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field


T = TypeVar("T", bound=BaseModel)


class MetadataSchema(BaseModel):
    title: str = ""
    keywords: list[str] = Field(default_factory=list)
    category: str = ""
    releases: str = ""


class TitleSchema(BaseModel):
    title: str = ""


class ReviewSchema(BaseModel):
    relevance_score: int = Field(default=0, ge=0, le=100)
    naturalness_score: int = Field(default=0, ge=0, le=100)
    keyword_accuracy_score: int = Field(default=0, ge=0, le=100)
    brand_logo_risk: bool = False
    model_release: str = "uncertain"
    property_release: str = "uncertain"
    issues: list[str] = Field(default_factory=list)
    notes: str = ""


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def model_to_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else None
    return None


def redact_secrets(value: Any, api_key: str | None = None) -> str:
    text = str(value or "")
    if api_key:
        text = text.replace(api_key, "[REDACTED_API_KEY]")
    text = re.sub(r"\bsk-[A-Za-z0-9_\-]{8,}\b", "[REDACTED_API_KEY]", text)
    text = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", text, flags=re.I)
    return text


def extract_response_text(response: Any) -> tuple[str, list[str]]:
    texts: list[str] = []
    refusals: list[str] = []

    output_text = get_attr(response, "output_text", "")
    if isinstance(output_text, str) and output_text.strip():
        texts.append(output_text.strip())

    output = get_attr(response, "output", []) or []
    for item in output:
        if get_attr(item, "type", "") != "message":
            continue
        for content in get_attr(item, "content", []) or []:
            content_type = get_attr(content, "type", "")
            if content_type == "output_text":
                text = get_attr(content, "text", "")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
            elif content_type == "refusal":
                refusal = get_attr(content, "refusal", "") or get_attr(content, "text", "")
                if refusal:
                    refusals.append(str(refusal))

    unique_texts = list(dict.fromkeys(texts))
    return "\n".join(unique_texts).strip(), refusals


def response_diagnostics(
    response: Any,
    model: str,
    token_limit: int,
    api_key: str | None = None,
) -> dict[str, Any]:
    raw_text, refusals = extract_response_text(response)
    incomplete = get_attr(response, "incomplete_details", None)
    error = get_attr(response, "error", None)
    output = get_attr(response, "output", []) or []

    return {
        "status": str(get_attr(response, "status", "unknown")),
        "incomplete_reason": get_attr(incomplete, "reason", None),
        "error_code": get_attr(error, "code", None),
        "error_message": redact_secrets(get_attr(error, "message", ""), api_key),
        "refusals": [redact_secrets(item, api_key) for item in refusals],
        "raw_output": redact_secrets(raw_text, api_key)[:12000],
        "model": model,
        "token_limit": token_limit,
        "output_types": [str(get_attr(item, "type", "unknown")) for item in output],
    }


def classify_api_error(exc: Exception, api_key: str | None = None) -> dict[str, str]:
    name = type(exc).__name__
    message = redact_secrets(str(exc), api_key)
    lowered = message.casefold()
    status_code = str(get_attr(exc, "status_code", ""))
    code = str(get_attr(get_attr(exc, "body", {}), "code", ""))

    if name == "AuthenticationError" or status_code == "401" or "invalid api key" in lowered:
        friendly = "API Key ไม่ถูกต้อง ถูกยกเลิก หรือหมดอายุ"
    elif (
        name == "NotFoundError"
        or status_code == "404"
        or "model_not_found" in lowered
        or "does not exist" in lowered
    ):
        friendly = "ไม่พบโมเดล หรือ API Key ไม่มีสิทธิ์ใช้โมเดลนี้"
    elif (
        "insufficient_quota" in lowered
        or "billing" in lowered
        or "quota" in lowered
        or "credit balance" in lowered
    ):
        friendly = "บัญชีไม่มีโควตา หรือยังไม่ได้ตั้งค่า Billing"
    elif name == "RateLimitError" or status_code == "429":
        friendly = "เรียก API ถี่เกินไปหรือชน Rate Limit กรุณาลดจำนวนงานพร้อมกันแล้วลองใหม่"
    elif "image" in lowered and any(
        token in lowered
        for token in ("unsupported", "not support", "invalid_image", "media type")
    ):
        friendly = "โมเดลนี้ไม่รองรับ Image Input หรือรูปแบบภาพที่ส่งไม่รองรับ"
    elif "max_output_tokens" in lowered:
        friendly = "คำตอบถูกตัดเนื่องจาก max_output_tokens ไม่เพียงพอ"
    elif name == "APIConnectionError":
        friendly = "เชื่อมต่อ OpenAI API ไม่สำเร็จ กรุณาตรวจอินเทอร์เน็ตหรือเครือข่าย"
    elif name in {"BadRequestError", "UnprocessableEntityError"} or status_code in {"400", "422"}:
        friendly = "คำขอไม่ถูกต้อง โมเดลนี้อาจไม่รองรับพารามิเตอร์หรือ Image Input ที่ใช้"
    else:
        friendly = "เกิดข้อผิดพลาดระหว่างเรียก OpenAI API"

    return {
        "friendly": friendly,
        "exception_type": name,
        "status_code": status_code,
        "code": code,
        "detail": message[:4000],
    }


def parse_json_direct(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def parse_json_block(raw_text: str) -> dict[str, Any] | None:
    for block in extract_balanced_json_objects(raw_text):
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_metadata_text(raw_text: str) -> dict[str, Any] | None:
    title_match = re.search(
        r"(?im)^\s*title\s*:\s*(.+?)(?=\n\s*(?:keywords?|category|releases?)\s*:|\Z)",
        raw_text,
        flags=re.S,
    )
    keywords_match = re.search(
        r"(?im)^\s*keywords?\s*:\s*(.+?)(?=\n\s*(?:title|category|releases?)\s*:|\Z)",
        raw_text,
        flags=re.S,
    )
    if not title_match and not keywords_match:
        return None

    payload: dict[str, Any] = {
        "title": title_match.group(1).strip() if title_match else "",
        "keywords": parse_keywords(keywords_match.group(1)) if keywords_match else [],
    }
    category_match = re.search(r"(?im)^\s*category\s*:\s*(.+)$", raw_text)
    releases_match = re.search(r"(?im)^\s*releases?\s*:\s*(.+)$", raw_text)
    if category_match:
        payload["category"] = category_match.group(1).strip()
    if releases_match:
        payload["releases"] = releases_match.group(1).strip()
    return payload


def parse_title_text(raw_text: str) -> dict[str, Any] | None:
    parsed = parse_json_direct(raw_text) or parse_json_block(raw_text)
    if parsed and parsed.get("title"):
        return {"title": str(parsed["title"]).strip()}
    match = re.search(r"(?im)^\s*title\s*:\s*(.+)$", raw_text)
    if match:
        return {"title": match.group(1).strip().strip('"')}
    single_line = raw_text.strip().strip('"`')
    if single_line and "\n" not in single_line and len(single_line) <= 300:
        return {"title": single_line}
    return None


def parse_review_text(raw_text: str) -> dict[str, Any] | None:
    parsed = parse_json_direct(raw_text) or parse_json_block(raw_text)
    return parsed if isinstance(parsed, dict) else None


def clean_metadata_payload(payload: dict[str, Any]) -> dict[str, Any]:
    title = str(payload.get("title") or "").strip().strip('"')
    keywords = parse_keywords(payload.get("keywords"))
    return {
        "title": title,
        "keywords": keywords,
        "category": str(payload.get("category") or "").strip(),
        "releases": str(payload.get("releases") or "").strip(),
    }


def metadata_payload_is_usable(payload: dict[str, Any]) -> bool:
    cleaned = clean_metadata_payload(payload)
    normalized = [re.sub(r"[^a-z0-9]+", " ", item.casefold()).strip() for item in cleaned["keywords"]]
    return (
        bool(cleaned["title"])
        and len(cleaned["keywords"]) == 49
        and len(set(normalized)) == 49
        and all(normalized)
    )


def title_payload_is_usable(payload: dict[str, Any]) -> bool:
    return bool(str(payload.get("title") or "").strip())


def review_payload_is_usable(payload: dict[str, Any]) -> bool:
    required = {"relevance_score", "naturalness_score", "keyword_accuracy_score"}
    return required.issubset(payload.keys())


def metadata_prompt(blacklist: list[str]) -> str:
    blacklist_text = ", ".join(sorted(set(blacklist))) if blacklist else "none supplied"
    return f"""
You are an expert Adobe Stock metadata editor. Analyze only what is visibly supported by the supplied image.

Return one JSON object with this exact shape:
{{
  "title": "English title",
  "keywords": ["keyword 1", "keyword 2", "... exactly 49 total"],
  "category": "",
  "releases": ""
}}

Work in this order:
1. Create exactly 49 distinct English keywords, ordered from most important to least important.
2. Keep the first 10 keywords short, concrete, commercially useful, and visually grounded. Multi-word phrases are allowed.
3. After the keyword list is final, write one natural English title of roughly 100–200 characters, never over 200 characters.
4. Every one of keywords 1–10 must appear in the title as the same exact word or contiguous phrase, ignoring capitalization only. Do not substitute synonyms.
5. The title must read naturally and must not look like a comma-separated keyword dump.

Hard rules:
- Do not begin with "Image of", "Photo of", or "Picture of".
- Do not infer location, nationality, ethnicity, profession, relationship, health status, age, identity, or intent unless visually unmistakable.
- Do not include brand names, logos, trademarks, celebrity names, copyrighted character names, camera brands, or stock-platform names.
- If a logo or brand mark is visible, describe the generic object instead and do not transcribe the brand.
- No duplicate keywords, including differences caused only by capitalization or punctuation.
- Avoid speculative concepts that are not supported by the visible scene.
- Do not include any of these blacklist terms as keywords: {blacklist_text}
- Output JSON only. Do not add Markdown fences or commentary.
""".strip()


def title_repair_prompt(title: str, top10: list[str]) -> str:
    return f"""
Rewrite the title below for Adobe Stock metadata.

Current title:
{title}

Required keywords, in priority order:
{json.dumps(top10, ensure_ascii=False)}

Rules:
- Use every required keyword exactly as written, as a contiguous phrase, ignoring capitalization only.
- Keep the meaning limited to the current title and required keywords. Do not add new visual facts.
- Write one natural English sentence, not a keyword dump.
- Maximum 200 characters.
- Do not begin with Image of, Photo of, or Picture of.
- Return JSON only: {{"title": "..."}}
""".strip()


def review_prompt(title: str, keywords: list[str]) -> str:
    return f"""
Review this Adobe Stock metadata against the supplied image. Judge only visible evidence.

Title:
{title}

Keywords:
{json.dumps(keywords, ensure_ascii=False)}

Return JSON only with this shape:
{{
  "relevance_score": 0,
  "naturalness_score": 0,
  "keyword_accuracy_score": 0,
  "brand_logo_risk": false,
  "model_release": "likely_required|not_required|uncertain",
  "property_release": "likely_required|not_required|uncertain",
  "issues": ["short issue"],
  "notes": "short note"
}}

Scoring is 0–100. Penalize unsupported claims, keyword stuffing, inaccurate keywords, visible branding risk, or unnatural English. Do not identify people or brands.
""".strip()


def build_image_input(prompt: str, data_url: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }
    ]


def build_text_input(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def call_structured(
    client: OpenAI,
    model: str,
    input_payload: list[dict[str, Any]],
    schema: type[T],
    token_limit: int,
    api_key: str,
    operation: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], Any]:
    response = client.responses.parse(
        model=model,
        store=False,
        input=input_payload,
        text_format=schema,
        max_output_tokens=token_limit,
    )
    diagnostics = response_diagnostics(response, model, token_limit, api_key)
    diagnostics["usage_event"] = build_usage_event(
        response,
        model=model,
        operation=operation,
        method="structured_output",
        token_limit=token_limit,
    )
    parsed = model_to_dict(get_attr(response, "output_parsed", None))
    return parsed, diagnostics, response


def call_plain(
    client: OpenAI,
    model: str,
    input_payload: list[dict[str, Any]],
    token_limit: int,
    api_key: str,
    operation: str,
) -> tuple[str, dict[str, Any], Any]:
    response = client.responses.create(
        model=model,
        store=False,
        input=input_payload,
        max_output_tokens=token_limit,
    )
    diagnostics = response_diagnostics(response, model, token_limit, api_key)
    diagnostics["usage_event"] = build_usage_event(
        response,
        model=model,
        operation=operation,
        method="plain_response",
        token_limit=token_limit,
    )
    raw_text, _ = extract_response_text(response)
    return raw_text, diagnostics, response


def _execute_hybrid_single_model(
    *,
    client: OpenAI,
    model: str,
    input_payload: list[dict[str, Any]],
    schema: type[T],
    fallback_parser: Callable[[str], dict[str, Any] | None],
    api_key: str,
    operation: str,
    data_validator: Callable[[dict[str, Any]], bool] | None = None,
    token_ladder: tuple[int, ...] = TOKEN_LADDER,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []

    def accepted(data: dict[str, Any] | None, method: str) -> bool:
        if not data:
            return False
        if data_validator is None or data_validator(data):
            return True
        attempts.append(
            {
                "method": method,
                "model": model,
                "validation_error": "Parsed output did not satisfy task-level metadata requirements",
            }
        )
        return False

    # Layer 1: Structured Output. Failure does not stop the workflow.
    first_limit = token_ladder[0]
    try:
        parsed, diagnostics, response = call_structured(
            client, model, input_payload, schema, first_limit, api_key, operation
        )
        diagnostics["method"] = "structured_output"
        attempts.append(diagnostics)
        if accepted(parsed, "structured_output_validation"):
            return {
                "ok": True,
                "data": parsed,
                "attempts": attempts,
                "method": "structured_output",
                "model_used": model,
            }

        raw_text, _ = extract_response_text(response)
        parsed_from_raw = (
            parse_json_direct(raw_text)
            or parse_json_block(raw_text)
            or fallback_parser(raw_text)
        )
        if accepted(parsed_from_raw, "structured_raw_validation"):
            return {
                "ok": True,
                "data": parsed_from_raw,
                "attempts": attempts,
                "method": "structured_raw_fallback",
                "model_used": model,
            }
    except Exception as exc:  # SDK/model compatibility varies; continue to plain response.
        attempts.append(
            {
                "method": "structured_output",
                "model": model,
                "token_limit": first_limit,
                "exception": classify_api_error(exc, api_key),
            }
        )

    # Layers 2–4: plain response -> direct JSON -> JSON block -> text parser.
    for token_limit in token_ladder:
        try:
            raw_text, diagnostics, response = call_plain(
                client, model, input_payload, token_limit, api_key, operation
            )
            diagnostics["method"] = "plain_response"
            attempts.append(diagnostics)

            parsed = (
                parse_json_direct(raw_text)
                or parse_json_block(raw_text)
                or fallback_parser(raw_text)
            )
            if accepted(parsed, "plain_response_validation"):
                return {
                    "ok": True,
                    "data": parsed,
                    "attempts": attempts,
                    "method": "plain_fallback",
                    "model_used": model,
                }

            reason = diagnostics.get("incomplete_reason")
            if reason != "max_output_tokens" and raw_text.strip():
                break
        except Exception as exc:
            error = classify_api_error(exc, api_key)
            attempts.append(
                {
                    "method": "plain_response",
                    "model": model,
                    "token_limit": token_limit,
                    "exception": error,
                }
            )
            # These errors will not improve by adding output tokens for the same model.
            if error["friendly"] in {
                "API Key ไม่ถูกต้อง ถูกยกเลิก หรือหมดอายุ",
                "บัญชีไม่มีโควตา หรือยังไม่ได้ตั้งค่า Billing",
                "ไม่พบโมเดล หรือ API Key ไม่มีสิทธิ์ใช้โมเดลนี้",
                "โมเดลนี้ไม่รองรับ Image Input หรือรูปแบบภาพที่ส่งไม่รองรับ",
                "คำขอไม่ถูกต้อง โมเดลนี้อาจไม่รองรับพารามิเตอร์หรือ Image Input ที่ใช้",
            }:
                break

    return {
        "ok": False,
        "data": None,
        "attempts": attempts,
        "method": "failed",
        "model_used": None,
        "error": "โมเดลตอบกลับมา แต่รูปแบบ Metadata ไม่ถูกต้อง หรือไม่มีผลลัพธ์ที่นำมาใช้ได้",
    }


def _has_global_fatal_error(attempts: list[dict[str, Any]]) -> bool:
    """Stop trying other models when the issue is account/network-wide."""
    fatal_messages = {
        "API Key ไม่ถูกต้อง ถูกยกเลิก หรือหมดอายุ",
        "บัญชีไม่มีโควตา หรือยังไม่ได้ตั้งค่า Billing",
        "เรียก API ถี่เกินไปหรือชน Rate Limit กรุณาลดจำนวนงานพร้อมกันแล้วลองใหม่",
        "เชื่อมต่อ OpenAI API ไม่สำเร็จ กรุณาตรวจอินเทอร์เน็ตหรือเครือข่าย",
    }
    for attempt in attempts:
        exception = attempt.get("exception") if isinstance(attempt, dict) else None
        if isinstance(exception, dict) and exception.get("friendly") in fatal_messages:
            return True
    return False


def normalize_model_candidates(model: str | Iterable[str]) -> list[str]:
    if isinstance(model, str):
        raw_models = [model]
    else:
        raw_models = [str(item) for item in model]
    return list(
        dict.fromkeys(
            item.strip()
            for item in raw_models
            if item and item.strip() and item.strip() != AUTO_MODEL_OPTION
        )
    )


def execute_hybrid(
    *,
    client: OpenAI,
    model: str | Iterable[str],
    input_payload: list[dict[str, Any]],
    schema: type[T],
    fallback_parser: Callable[[str], dict[str, Any] | None],
    api_key: str,
    operation: str = "api_call",
    data_validator: Callable[[dict[str, Any]], bool] | None = None,
    token_ladder: tuple[int, ...] = TOKEN_LADDER,
) -> dict[str, Any]:
    """Try the selected model, then automatically fall back to compatible models."""
    candidates = normalize_model_candidates(model)[:6]
    if not candidates:
        candidates = list(MODEL_DISCOVERY_FAILURE_CHAIN)[:6]

    all_attempts: list[dict[str, Any]] = []
    models_tried: list[str] = []
    last_error = "ไม่พบโมเดลที่ใช้งานได้"

    for candidate in candidates:
        models_tried.append(candidate)
        result = _execute_hybrid_single_model(
            client=client,
            model=candidate,
            input_payload=input_payload,
            schema=schema,
            fallback_parser=fallback_parser,
            api_key=api_key,
            operation=operation,
            data_validator=data_validator,
            token_ladder=token_ladder,
        )
        all_attempts.extend(result.get("attempts", []))
        if result.get("ok"):
            result["attempts"] = all_attempts
            result["models_tried"] = models_tried
            result["fallback_used"] = len(models_tried) > 1
            result["usage_summary"] = summarize_usage_events(
                usage_events_from_attempts(all_attempts)
            )
            return result

        last_error = str(result.get("error") or last_error)
        if _has_global_fatal_error(result.get("attempts", [])):
            break

    return {
        "ok": False,
        "data": None,
        "attempts": all_attempts,
        "method": "failed",
        "model_used": None,
        "models_tried": models_tried,
        "fallback_used": len(models_tried) > 1,
        "error": last_error,
        "usage_summary": summarize_usage_events(
            usage_events_from_attempts(all_attempts)
        ),
    }


def _model_rank_key(model_id: str) -> tuple[int, int, str]:
    lowered = model_id.casefold()

    # Exact aliases first, so gpt-4.1-mini is not accidentally ranked as gpt-4.1.
    for index, preferred in enumerate(MODEL_QUALITY_PRIORITY):
        if lowered == preferred.casefold():
            return (index, 0, lowered)

    for index, preferred in enumerate(MODEL_QUALITY_PRIORITY):
        preferred_lower = preferred.casefold()
        if lowered.startswith(preferred_lower + "-"):
            # Prefer stable aliases over dated snapshots / derivative variants.
            penalty = 4
            if "mini" in lowered:
                penalty += 20
            if "nano" in lowered:
                penalty += 40
            if "preview" in lowered:
                penalty += 10
            return (index, penalty, lowered)

    family_penalty = 900
    if lowered.startswith("gpt-5"):
        family_penalty = 500
    elif lowered.startswith(("gpt-4.1", "gpt-4o", "gpt-4.5")):
        family_penalty = 600
    elif lowered.startswith(("o4", "o3", "o1")):
        family_penalty = 700
    return (family_penalty, 0, lowered)


def rank_models(models: Iterable[str]) -> list[str]:
    cleaned = list(dict.fromkeys(str(model).strip() for model in models if str(model).strip()))
    return sorted(cleaned, key=_model_rank_key)


def discover_models(api_key: str) -> list[str]:
    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=2)
    page = client.models.list()
    model_ids = {
        str(get_attr(item, "id", "")).strip()
        for item in (get_attr(page, "data", []) or [])
        if str(get_attr(item, "id", "")).strip()
    }

    candidates = [
        model_id
        for model_id in model_ids
        if not any(token in model_id.casefold() for token in OBVIOUS_NON_VISION_MODEL_TOKENS)
        and model_id.casefold().startswith(("gpt-", "o1", "o3", "o4"))
    ]
    return rank_models(candidates or model_ids)


def choose_default_model(models: list[str]) -> str:
    ranked = rank_models(models)
    return ranked[0] if ranked else DEFAULT_MODEL_ID


def build_model_fallback_chain(
    available_models: list[str] | None,
    selected_model: str = AUTO_MODEL_OPTION,
    auto_fallback: bool = True,
) -> list[str]:
    """Build a safe chain without assuming that any unlisted model exists."""
    ranked_available = rank_models(available_models or [])
    discovered = bool(ranked_available)
    base_chain = (ranked_available[:6] if discovered else list(MODEL_DISCOVERY_FAILURE_CHAIN)[:6])

    selected = (selected_model or AUTO_MODEL_OPTION).strip()
    if selected == AUTO_MODEL_OPTION:
        chain = list(base_chain)
    else:
        chain = [selected]
        if auto_fallback:
            chain.extend(base_chain)

    # Keep the old reliable vision model as a last-resort compatibility fallback
    # only when discovery failed or the account explicitly exposes it.
    if auto_fallback and (
        not discovered or LEGACY_VISION_FALLBACK_MODEL_ID in ranked_available
    ):
        chain.append(LEGACY_VISION_FALLBACK_MODEL_ID)

    return list(dict.fromkeys(model for model in chain if model and model != AUTO_MODEL_OPTION))


def analyze_image(
    *,
    original_bytes: bytes,
    image_sha256: str,
    api_key: str,
    model: str | Iterable[str],
    blacklist: list[str],
    use_cache: bool = True,
    auto_repair_title: bool = True,
) -> dict[str, Any]:
    model_candidates = normalize_model_candidates(model)
    model_signature = json.dumps(model_candidates, ensure_ascii=False)
    cache_file = cache_path(
        ANALYSIS_CACHE_DIR,
        "metadata",
        image_sha256,
        model_signature,
        json.dumps(sorted(blacklist), ensure_ascii=False),
    )
    if use_cache:
        cached = read_json(cache_file)
        if cached and cached.get("ok"):
            cached["from_cache"] = True
            return cached

    data_url, analysis_info = create_analysis_data_url(original_bytes)
    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=2)
    result = execute_hybrid(
        client=client,
        model=model,
        input_payload=build_image_input(metadata_prompt(blacklist), data_url),
        schema=MetadataSchema,
        fallback_parser=parse_metadata_text,
        api_key=api_key,
        operation="generate_metadata",
        data_validator=metadata_payload_is_usable,
    )

    if not result.get("ok"):
        result["analysis_info"] = analysis_info
        return result

    cleaned = clean_metadata_payload(result["data"])
    result["data"] = cleaned
    result["analysis_info"] = analysis_info
    result["from_cache"] = False

    if auto_repair_title and len(cleaned["keywords"]) >= 10:
        coverage = top_keyword_coverage(cleaned["title"], cleaned["keywords"])
        if coverage["count"] < 10 or len(cleaned["title"]) > 200:
            successful_model = str(result.get("model_used") or "").strip()
            repair_chain = list(dict.fromkeys(
                ([successful_model] if successful_model else []) + model_candidates
            ))
            repaired = regenerate_title(
                api_key=api_key,
                model=repair_chain,
                title=cleaned["title"],
                keywords=cleaned["keywords"],
                max_attempts=2,
                use_cache=use_cache,
                operation="auto_title_repair",
            )
            result["title_repair"] = repaired
            if repaired.get("ok") and repaired.get("title"):
                cleaned["title"] = repaired["title"]

    write_json_atomic(cache_file, result)
    return result


def regenerate_title(
    *,
    api_key: str,
    model: str | Iterable[str],
    title: str,
    keywords: list[str],
    max_attempts: int = TITLE_REPAIR_MAX_ATTEMPTS,
    use_cache: bool = True,
    operation: str = "title_repair",
) -> dict[str, Any]:
    keywords = parse_keywords(keywords)
    top10 = keywords[:10]
    if len(top10) < 10:
        return {
            "ok": False,
            "title": title,
            "missing": top10,
            "error": "ต้องมี Keywords อย่างน้อย 10 คำก่อนสร้าง Title ใหม่",
            "attempts": [],
            "usage_summary": summarize_usage_events([]),
        }

    model_candidates = normalize_model_candidates(model)
    model_signature = json.dumps(model_candidates, ensure_ascii=False)
    cache_file = cache_path(
        TITLE_CACHE_DIR,
        "title",
        model_signature,
        title,
        json.dumps(top10, ensure_ascii=False),
    )
    if use_cache:
        cached = read_json(cache_file)
        if cached and cached.get("ok"):
            cached["from_cache"] = True
            return cached

    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=2)
    current_title = title.strip()
    all_attempts: list[dict[str, Any]] = []
    all_models_tried: list[str] = []
    current_model_chain = list(model_candidates)

    for repair_index in range(max(1, min(max_attempts, TITLE_REPAIR_MAX_ATTEMPTS))):
        hybrid = execute_hybrid(
            client=client,
            model=current_model_chain,
            input_payload=build_text_input(title_repair_prompt(current_title, top10)),
            schema=TitleSchema,
            fallback_parser=parse_title_text,
            api_key=api_key,
            operation=operation,
            data_validator=title_payload_is_usable,
            token_ladder=(1000, 1600, 2400),
        )
        all_attempts.extend(hybrid.get("attempts", []))
        all_models_tried.extend(hybrid.get("models_tried", []))
        successful_model = str(hybrid.get("model_used") or "").strip()
        if successful_model:
            current_model_chain = list(dict.fromkeys([successful_model] + current_model_chain))
        if not hybrid.get("ok"):
            break

        candidate = str((hybrid.get("data") or {}).get("title") or "").strip().strip('"')
        if candidate:
            current_title = candidate
        coverage = top_keyword_coverage(current_title, top10)
        if coverage["count"] == 10 and len(current_title) <= 200:
            output = {
                "ok": True,
                "title": current_title,
                "missing": [],
                "coverage": 10,
                "attempts": all_attempts,
                "repair_rounds": repair_index + 1,
                "model_used": hybrid.get("model_used"),
                "models_tried": list(dict.fromkeys(all_models_tried)),
                "fallback_used": bool(hybrid.get("fallback_used")),
                "from_cache": False,
                "usage_summary": summarize_usage_events(
                    usage_events_from_attempts(all_attempts)
                ),
            }
            write_json_atomic(cache_file, output)
            return output

    coverage = top_keyword_coverage(current_title, top10)
    output = {
        "ok": False,
        "title": current_title,
        "missing": coverage["missing"],
        "coverage": coverage["count"],
        "attempts": all_attempts,
        "repair_rounds": min(max_attempts, TITLE_REPAIR_MAX_ATTEMPTS),
        "model_used": None,
        "models_tried": list(dict.fromkeys(all_models_tried)),
        "fallback_used": len(list(dict.fromkeys(all_models_tried))) > 1,
        "error": "สร้าง Title ใหม่แล้ว แต่ยังไม่ผ่านเงื่อนไขครบทุกข้อ",
        "from_cache": False,
        "usage_summary": summarize_usage_events(
            usage_events_from_attempts(all_attempts)
        ),
    }
    write_json_atomic(cache_file, output)
    return output


def review_metadata(
    *,
    original_bytes: bytes,
    image_sha256: str,
    api_key: str,
    model: str | Iterable[str],
    title: str,
    keywords: list[str],
    use_cache: bool = True,
) -> dict[str, Any]:
    keywords = parse_keywords(keywords)
    model_candidates = normalize_model_candidates(model)
    model_signature = json.dumps(model_candidates, ensure_ascii=False)
    cache_file = cache_path(
        QUALITY_CACHE_DIR,
        "review",
        image_sha256,
        model_signature,
        title,
        json.dumps(keywords, ensure_ascii=False),
    )
    if use_cache:
        cached = read_json(cache_file)
        if cached and cached.get("ok"):
            cached["from_cache"] = True
            return cached

    data_url, _ = create_analysis_data_url(original_bytes)
    client = OpenAI(api_key=api_key, timeout=120.0, max_retries=2)
    hybrid = execute_hybrid(
        client=client,
        model=model,
        input_payload=build_image_input(review_prompt(title, keywords), data_url),
        schema=ReviewSchema,
        fallback_parser=parse_review_text,
        api_key=api_key,
        operation="ai_review",
        data_validator=review_payload_is_usable,
        token_ladder=(1400, 2200, 3200),
    )

    if not hybrid.get("ok"):
        return hybrid

    try:
        review = ReviewSchema.model_validate(hybrid["data"]).model_dump()
    except Exception:
        raw = hybrid.get("data") or {}
        review = {
            "relevance_score": int(raw.get("relevance_score", 0) or 0),
            "naturalness_score": int(raw.get("naturalness_score", 0) or 0),
            "keyword_accuracy_score": int(raw.get("keyword_accuracy_score", 0) or 0),
            "brand_logo_risk": bool(raw.get("brand_logo_risk", False)),
            "model_release": str(raw.get("model_release", "uncertain")),
            "property_release": str(raw.get("property_release", "uncertain")),
            "issues": [str(item) for item in (raw.get("issues") or [])],
            "notes": str(raw.get("notes", "")),
        }
        for score_key in ("relevance_score", "naturalness_score", "keyword_accuracy_score"):
            review[score_key] = max(0, min(100, int(review[score_key])))

    output = {
        "ok": True,
        "review": review,
        "attempts": hybrid.get("attempts", []),
        "method": hybrid.get("method"),
        "model_used": hybrid.get("model_used"),
        "models_tried": hybrid.get("models_tried", []),
        "fallback_used": bool(hybrid.get("fallback_used")),
        "from_cache": False,
        "usage_summary": summarize_usage_events(
            usage_events_from_attempts(hybrid.get("attempts", []))
        ),
    }
    write_json_atomic(cache_file, output)
    return output

# ========================================================================================
# Inlined from: app.py
# ========================================================================================

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import streamlit as st


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 4rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-muted {color: #667085; font-size: 0.9rem;}
.status-card {padding: 0.8rem 1rem; border: 1px solid #e5e7eb; border-radius: 12px; margin: 0.35rem 0;}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    # Session state can survive reruns during a Cloud process lifetime. Keep defaults
    # simple and repair obviously incompatible values after dependency upgrades.
    defaults: dict[str, Any] = {
        "assets": {},
        "asset_order": [],
        "uploader_nonce": 0,
        "available_models": [],
        "selected_model_id": AUTO_MODEL_OPTION,
        "export_bundle": None,
        "excluded_exports": [],
        "flash": None,
        "usd_to_thb": DEFAULT_USD_TO_THB,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Defensive repair for stale/corrupted state after Cloud rebuilds.
    if not isinstance(st.session_state.get("assets"), dict):
        st.session_state.assets = {}
    if not isinstance(st.session_state.get("asset_order"), list):
        st.session_state.asset_order = []
    if not isinstance(st.session_state.get("available_models"), list):
        st.session_state.available_models = []


def set_flash(level: str, message: str) -> None:
    st.session_state.flash = {"level": level, "message": message}


def show_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if not flash:
        return
    level = flash.get("level", "info")
    message = flash.get("message", "")
    getattr(st, level, st.info)(message)


def secret_from_streamlit() -> str:
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "") or "").strip()
    except Exception:
        return ""


def resolve_api_key(manual_key: str) -> tuple[str, str]:
    manual_key = (manual_key or "").strip()
    if manual_key:
        return manual_key, "Manual Input"
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key, "Environment Variable"
    secret_key = secret_from_streamlit()
    if secret_key:
        return secret_key, "Streamlit Secrets"
    return "", "ไม่พบ API Key"


def parse_blacklist(text: str) -> list[str]:
    items = parse_keywords(text)
    return list(dict.fromkeys(item.casefold().strip() for item in items if item.strip()))


def sync_uploaded_files(uploaded_files: list[Any]) -> None:
    active_assets: dict[str, dict[str, Any]] = {}
    active_order: list[str] = []
    existing = st.session_state.assets

    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        digest = sha256_bytes(data)
        name_digest = sha256_bytes(uploaded.name.encode("utf-8"))[:8]
        asset_id = f"{digest[:20]}_{name_digest}_{Path(uploaded.name).suffix.casefold()}"
        active_order.append(asset_id)

        if asset_id in existing:
            record = existing[asset_id]
            record["filename"] = uploaded.name
            active_assets[asset_id] = record
            continue

        try:
            image_info = inspect_image(data)
            error = ""
        except Exception as exc:
            image_info = {
                "width": 0,
                "height": 0,
                "mode": "unknown",
                "format": "unknown",
                "transparency": False,
                "dpi": None,
                "has_icc_profile": False,
            }
            error = str(exc)

        active_assets[asset_id] = {
            "id": asset_id,
            "filename": uploaded.name,
            "bytes": data,
            "sha256": digest,
            "size_bytes": len(data),
            "mime_type": uploaded.type or "application/octet-stream",
            "image_info": image_info,
            "title": "",
            "keywords": [],
            "category": "",
            "releases": "",
            "analysis_status": "ยังไม่ได้วิเคราะห์" if not error else "ไฟล์มีปัญหา",
            "error": error,
            "diagnostics": [],
            "ai_review": None,
            "quality": None,
            "from_cache": False,
            "model_used": "",
            "models_tried": [],
            "fallback_used": False,
            "review_model_used": "",
            "api_usage_events": [],
            "api_usage_summary": summarize_usage_events([]),
            "cache_usage_reference": [],
            "ui_revision": 0,
        }

    if active_order != st.session_state.asset_order:
        st.session_state.export_bundle = None
    st.session_state.assets = active_assets
    st.session_state.asset_order = active_order


def friendly_result_error(result: dict[str, Any]) -> str:
    for attempt in reversed(result.get("attempts", [])):
        exception = attempt.get("exception") if isinstance(attempt, dict) else None
        if isinstance(exception, dict) and exception.get("friendly"):
            detail = exception.get("detail", "")
            return f"{exception['friendly']}\n\nรายละเอียด: {detail}" if detail else exception["friendly"]
        if isinstance(attempt, dict) and attempt.get("error_message"):
            return str(attempt["error_message"])
        if isinstance(attempt, dict) and attempt.get("incomplete_reason") == "max_output_tokens":
            return "คำตอบถูกตัดเนื่องจาก max_output_tokens ไม่เพียงพอ"
        refusals = attempt.get("refusals", []) if isinstance(attempt, dict) else []
        if refusals:
            return "โมเดลปฏิเสธคำขอ: " + " | ".join(str(item) for item in refusals)
    return str(result.get("error") or "ไม่สามารถสร้าง Metadata ได้")


def register_result_usage(
    record: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    usd_to_thb: float,
    include_title_repair: bool = False,
) -> None:
    if not result:
        return

    if result.get("from_cache"):
        reference = result.get("usage_summary")
        if isinstance(reference, dict) and reference.get("api_calls"):
            record.setdefault("cache_usage_reference", []).append(reference)
    else:
        events = usage_events_from_attempts(result.get("attempts", []))
        record["api_usage_events"] = merge_usage_events(
            record.get("api_usage_events", []),
            events,
        )

    if include_title_repair and not result.get("from_cache"):
        nested = result.get("title_repair")
        if isinstance(nested, dict):
            register_result_usage(
                record,
                nested,
                usd_to_thb=usd_to_thb,
                include_title_repair=False,
            )

    record["api_usage_summary"] = summarize_usage_events(
        record.get("api_usage_events", []),
        usd_to_thb,
    )


def format_usd(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):.6f}"


def render_api_cost(record: dict[str, Any], usd_to_thb: float) -> None:
    summary = summarize_usage_events(record.get("api_usage_events", []), usd_to_thb)
    record["api_usage_summary"] = summary

    st.subheader("API Usage & Cost")
    cols = st.columns(5)
    cols[0].metric("API Calls", summary["api_calls"])
    cols[1].metric("Input Tokens", f"{summary['input_tokens']:,}")
    cols[2].metric("Cached Input", f"{summary['cached_input_tokens']:,}")
    cols[3].metric("Output Tokens", f"{summary['output_tokens']:,}")
    cols[4].metric(
        "Cost / ภาพ",
        format_usd(summary["known_cost_usd"]),
        f"≈ ฿{summary['estimated_cost_thb']:.4f}",
    )

    if summary["unknown_pricing_models"]:
        st.warning(
            "มี Token Usage แต่ยังไม่รวมราคาโมเดล: "
            + ", ".join(summary["unknown_pricing_models"])
            + " — ระบบจะไม่เดาราคา"
        )
    if record.get("from_cache") and not record.get("api_usage_events"):
        st.info("ผลครั้งนี้มาจาก App Cache จึงไม่มีค่า API เพิ่มใน Session นี้")
    if record.get("cache_usage_reference"):
        previous = record["cache_usage_reference"][-1]
        st.caption(
            "ผลที่อ่านจาก Cache มีประวัติเดิมประมาณ "
            f"{previous.get('api_calls', 0)} API calls / "
            f"{format_usd(previous.get('known_cost_usd', 0.0))} "
            "แต่ไม่ได้ถูกคิดซ้ำใน Session ปัจจุบัน"
        )

    events = record.get("api_usage_events", [])
    if events:
        rows: list[dict[str, Any]] = []
        for event in events:
            usage = event.get("usage") or {}
            cost = event.get("cost") or {}
            rows.append(
                {
                    "Operation": event.get("operation", ""),
                    "Model": event.get("model", ""),
                    "Method": event.get("method", ""),
                    "Input": usage.get("input_tokens", 0),
                    "Cached": usage.get("cached_input_tokens", 0),
                    "Output": usage.get("output_tokens", 0),
                    "Cost USD": (
                        round(float(cost["total_cost_usd"]), 8)
                        if cost.get("total_cost_usd") is not None
                        else None
                    ),
                }
            )
        with st.expander("ดูรายละเอียดค่า API ทุกครั้งที่เรียก"):
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                f"ราคาอ้างอิงในโค้ดอัปเดต {PRICING_UPDATED_AT}; "
                "ราคาเงินบาทเป็นค่าประมาณตามอัตราที่ตั้งไว้ใน Sidebar"
            )


def recompute_quality(record: dict[str, Any], blacklist: list[str], ai_threshold: int) -> None:
    record["keywords"] = parse_keywords(record.get("keywords"))
    record["quality"] = evaluate_quality(
        record.get("title", ""),
        record["keywords"],
        blacklist,
        record.get("ai_review"),
        ai_threshold,
    )


def apply_analysis_result(
    record: dict[str, Any],
    result: dict[str, Any],
    blacklist: list[str],
    ai_threshold: int,
    usd_to_thb: float,
) -> None:
    register_result_usage(
        record, result, usd_to_thb=usd_to_thb, include_title_repair=True
    )
    record["diagnostics"] = result.get("attempts", [])
    record["from_cache"] = bool(result.get("from_cache"))
    record["model_used"] = str(result.get("model_used") or "")
    record["models_tried"] = list(result.get("models_tried") or [])
    record["fallback_used"] = bool(result.get("fallback_used"))

    if not result.get("ok"):
        record["analysis_status"] = "วิเคราะห์ไม่สำเร็จ"
        record["error"] = friendly_result_error(result)
        recompute_quality(record, blacklist, ai_threshold)
        return

    data = result.get("data") or {}
    record["title"] = str(data.get("title") or "").strip()
    record["keywords"] = parse_keywords(data.get("keywords"))
    record["category"] = str(data.get("category") or "").strip()
    record["releases"] = str(data.get("releases") or "").strip()
    record["analysis_status"] = "วิเคราะห์แล้ว"
    record["error"] = ""
    record["ui_revision"] += 1
    recompute_quality(record, blacklist, ai_threshold)
    st.session_state.export_bundle = None


def apply_review_result(
    record: dict[str, Any],
    result: dict[str, Any],
    blacklist: list[str],
    ai_threshold: int,
    usd_to_thb: float,
) -> None:
    register_result_usage(record, result, usd_to_thb=usd_to_thb)
    if result.get("ok"):
        record["ai_review"] = result.get("review") or {}
        record["review_diagnostics"] = result.get("attempts", [])
        record["review_model_used"] = str(result.get("model_used") or "")
        record["error"] = ""
    else:
        record["error"] = friendly_result_error(result)
        record["review_diagnostics"] = result.get("attempts", [])
    recompute_quality(record, blacklist, ai_threshold)
    st.session_state.export_bundle = None


def analyze_worker(
    asset_id: str,
    record: dict[str, Any],
    *,
    api_key: str,
    model: list[str],
    blacklist: list[str],
    use_cache: bool,
    auto_review: bool,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    analysis_result = analyze_image(
        original_bytes=record["bytes"],
        image_sha256=record["sha256"],
        api_key=api_key,
        model=model,
        blacklist=blacklist,
        use_cache=use_cache,
        auto_repair_title=True,
    )

    review_result: dict[str, Any] | None = None
    if auto_review and analysis_result.get("ok"):
        data = analysis_result.get("data") or {}
        review_result = review_metadata(
            original_bytes=record["bytes"],
            image_sha256=record["sha256"],
            api_key=api_key,
            model=model,
            title=str(data.get("title") or ""),
            keywords=parse_keywords(data.get("keywords")),
            use_cache=use_cache,
        )
    return asset_id, analysis_result, review_result


def run_batch_analysis(
    asset_ids: list[str],
    *,
    api_key: str,
    model: list[str],
    blacklist: list[str],
    use_cache: bool,
    auto_review: bool,
    max_workers: int,
    ai_threshold: int,
    usd_to_thb: float,
) -> None:
    progress = st.progress(0, text="กำลังเตรียมวิเคราะห์ภาพ...")
    completed = 0
    success_count = 0

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 4))) as executor:
        futures = {
            executor.submit(
                analyze_worker,
                asset_id,
                dict(st.session_state.assets[asset_id]),
                api_key=api_key,
                model=model,
                blacklist=blacklist,
                use_cache=use_cache,
                auto_review=auto_review,
            ): asset_id
            for asset_id in asset_ids
        }

        for future in as_completed(futures):
            asset_id = futures[future]
            record = st.session_state.assets[asset_id]
            try:
                _, analysis_result, review_result = future.result()
                apply_analysis_result(record, analysis_result, blacklist, ai_threshold, usd_to_thb)
                if review_result is not None:
                    apply_review_result(record, review_result, blacklist, ai_threshold, usd_to_thb)
                if analysis_result.get("ok"):
                    success_count += 1
            except Exception as exc:
                error = classify_api_error(exc, api_key)
                record["analysis_status"] = "วิเคราะห์ไม่สำเร็จ"
                record["error"] = f"{error['friendly']}\n\nรายละเอียด: {error['detail']}"
                recompute_quality(record, blacklist, ai_threshold)

            completed += 1
            progress.progress(
                completed / len(asset_ids),
                text=f"วิเคราะห์แล้ว {completed}/{len(asset_ids)} ภาพ",
            )

    progress.empty()
    set_flash("success", f"วิเคราะห์สำเร็จ {success_count}/{len(asset_ids)} ภาพ")
    st.rerun()


def status_message(quality: dict[str, Any] | None) -> None:
    if not quality:
        st.info("ยังไม่มีผลตรวจคุณภาพ")
        return
    status = quality.get("status", "")
    message = f"{status} — คะแนน {quality.get('score', 0)}/100"
    if status.startswith("ผ่าน"):
        st.success(message)
    elif status.startswith("ไม่ผ่าน"):
        st.error(message)
    else:
        st.warning(message)


def render_ai_review(review: dict[str, Any] | None) -> None:
    if not review:
        st.caption("ยังไม่ได้ AI Review")
        return
    cols = st.columns(3)
    cols[0].metric("Image relevance", review.get("relevance_score", 0))
    cols[1].metric("Title naturalness", review.get("naturalness_score", 0))
    cols[2].metric("Keyword accuracy", review.get("keyword_accuracy_score", 0))
    st.caption(
        f"Brand/Logo risk: {'พบความเสี่ยง' if review.get('brand_logo_risk') else 'ไม่พบชัดเจน'} · "
        f"Model release: {review.get('model_release', 'uncertain')} · "
        f"Property release: {review.get('property_release', 'uncertain')}"
    )
    if review.get("issues"):
        st.warning("AI Issues: " + " | ".join(str(item) for item in review["issues"]))
    if review.get("notes"):
        st.caption(str(review["notes"]))


def render_asset(
    record: dict[str, Any],
    *,
    api_key: str,
    model: list[str],
    blacklist: list[str],
    use_cache: bool,
    ai_threshold: int,
    show_debug: bool,
    usd_to_thb: float,
) -> None:
    asset_id = record["id"]
    revision = record.get("ui_revision", 0)
    info = record["image_info"]

    with st.expander(f"{record['filename']} — {record['analysis_status']}", expanded=True):
        preview_col, form_col = st.columns([0.9, 1.8], gap="large")

        with preview_col:
            st.image(record["bytes"], use_container_width=True)
            st.markdown(
                f"""
<div class="small-muted">
<b>ไฟล์:</b> {record['filename']}<br>
<b>ขนาด:</b> {human_bytes(record['size_bytes'])}<br>
<b>มิติ:</b> {info['width']} × {info['height']} px<br>
<b>Mode:</b> {info['mode']} · <b>Format:</b> {info['format']}<br>
<b>Transparency:</b> {'Yes' if info['transparency'] else 'No'}<br>
<b>ICC Profile:</b> {'Yes' if info['has_icc_profile'] else 'No'}
</div>
""",
                unsafe_allow_html=True,
            )
            if record.get("from_cache"):
                st.caption("ผลวิเคราะห์นี้โหลดจาก Cache")
            if record.get("model_used"):
                st.caption(f"โมเดลที่สร้าง Metadata จริง: {record['model_used']}")
            if record.get("fallback_used") and record.get("models_tried"):
                st.info("มีการสลับโมเดลอัตโนมัติ: " + " → ".join(record["models_tried"]))

        with form_col:
            title_key = f"title_{asset_id}_{revision}"
            keywords_key = f"keywords_{asset_id}_{revision}"
            category_key = f"category_{asset_id}_{revision}"
            releases_key = f"releases_{asset_id}_{revision}"

            new_title = st.text_area(
                "Title",
                value=record.get("title", ""),
                height=100,
                key=title_key,
                help="ต้องไม่เกิน 200 ตัวอักษร และต้องมี Keywords 10 คำแรกครบ",
            )
            new_keywords_text = st.text_area(
                "Keywords — คั่นด้วย comma หรือขึ้นบรรทัดใหม่",
                value=", ".join(parse_keywords(record.get("keywords"))),
                height=150,
                key=keywords_key,
            )
            small_cols = st.columns(2)
            new_category = small_cols[0].text_input(
                "Category",
                value=record.get("category", ""),
                key=category_key,
            )
            new_releases = small_cols[1].text_input(
                "Releases",
                value=record.get("releases", ""),
                key=releases_key,
            )

            parsed_new_keywords = parse_keywords(new_keywords_text)
            changed = (
                new_title != record.get("title", "")
                or parsed_new_keywords != parse_keywords(record.get("keywords"))
                or new_category != record.get("category", "")
                or new_releases != record.get("releases", "")
            )
            record["title"] = new_title
            record["keywords"] = parsed_new_keywords
            record["category"] = new_category
            record["releases"] = new_releases
            if changed:
                st.session_state.export_bundle = None
            recompute_quality(record, blacklist, ai_threshold)

            quality = record["quality"]
            metrics = st.columns(4)
            metrics[0].metric("Keywords", f"{quality['keyword_count']}/49")
            metrics[1].metric("Top 10 Coverage", f"{quality['top10_coverage']}/10")
            metrics[2].metric("Title Length", f"{quality['title_length']}/200")
            metrics[3].metric("Quality Score", f"{quality['score']}/100")
            status_message(quality)

            if quality["missing_keywords"]:
                st.error("Missing Keywords: " + ", ".join(quality["missing_keywords"]))
            if quality["issues"]:
                st.caption("Issues: " + " | ".join(quality["issues"]))
            if quality["warnings"]:
                st.caption("Warnings: " + " | ".join(quality["warnings"]))

            action_cols = st.columns(4)
            analyze_clicked = action_cols[0].button(
                "วิเคราะห์ภาพนี้",
                key=f"analyze_{asset_id}",
                use_container_width=True,
            )
            regenerate_clicked = action_cols[1].button(
                "สร้าง Title ใหม่",
                key=f"regenerate_{asset_id}",
                use_container_width=True,
            )
            check_clicked = action_cols[2].button(
                "ตรวจด้วยกฎ",
                key=f"rulecheck_{asset_id}",
                use_container_width=True,
            )
            ai_review_clicked = action_cols[3].button(
                "AI Review",
                key=f"aireview_{asset_id}",
                use_container_width=True,
            )

            if analyze_clicked:
                if not api_key:
                    st.error("กรุณาใส่ OpenAI API Key ก่อนวิเคราะห์")
                elif record.get("error") and info.get("width", 0) == 0:
                    st.error(record["error"])
                else:
                    with st.spinner("กำลังวิเคราะห์ภาพและสร้าง Metadata..."):
                        result = analyze_image(
                            original_bytes=record["bytes"],
                            image_sha256=record["sha256"],
                            api_key=api_key,
                            model=model,
                            blacklist=blacklist,
                            use_cache=use_cache,
                            auto_repair_title=True,
                        )
                    apply_analysis_result(record, result, blacklist, ai_threshold, usd_to_thb)
                    set_flash("success" if result.get("ok") else "error", "วิเคราะห์ภาพเสร็จแล้ว" if result.get("ok") else record["error"])
                    st.rerun()

            if regenerate_clicked:
                if not api_key:
                    st.error("กรุณาใส่ OpenAI API Key ก่อนสร้าง Title ใหม่")
                else:
                    with st.spinner("กำลังเรียบเรียง Title ใหม่จาก Keywords 10 คำแรก..."):
                        result = regenerate_title(
                            api_key=api_key,
                            model=model,
                            title=record.get("title", ""),
                            keywords=record.get("keywords", []),
                            max_attempts=3,
                            use_cache=use_cache,
                            operation="manual_title_regeneration",
                        )
                    register_result_usage(record, result, usd_to_thb=usd_to_thb)
                    record["diagnostics"] = result.get("attempts", [])
                    if result.get("model_used"):
                        record["model_used"] = str(result.get("model_used"))
                    record["models_tried"] = list(result.get("models_tried") or record.get("models_tried") or [])
                    record["fallback_used"] = bool(result.get("fallback_used"))
                    if result.get("title"):
                        record["title"] = result["title"]
                        record["ui_revision"] += 1
                    record["error"] = "" if result.get("ok") else str(result.get("error") or "")
                    recompute_quality(record, blacklist, ai_threshold)
                    st.session_state.export_bundle = None
                    if result.get("ok"):
                        set_flash("success", "สร้าง Title ใหม่สำเร็จ และ Top 10 Coverage ครบ 10/10")
                    else:
                        missing = ", ".join(result.get("missing", []))
                        set_flash("warning", f"ยังไม่ผ่าน คำที่ขาด: {missing or 'ไม่ทราบ'}")
                    st.rerun()

            if check_clicked:
                recompute_quality(record, blacklist, ai_threshold)
                set_flash("success" if record["quality"]["hard_pass"] else "warning", record["quality"]["status"])
                st.rerun()

            if ai_review_clicked:
                if not api_key:
                    st.error("กรุณาใส่ OpenAI API Key ก่อนใช้ AI Review")
                elif not record.get("title") or not record.get("keywords"):
                    st.error("ต้องมี Title และ Keywords ก่อนใช้ AI Review")
                else:
                    with st.spinner("กำลังตรวจความตรงของ Metadata กับภาพ..."):
                        result = review_metadata(
                            original_bytes=record["bytes"],
                            image_sha256=record["sha256"],
                            api_key=api_key,
                            model=model,
                            title=record["title"],
                            keywords=record["keywords"],
                            use_cache=use_cache,
                        )
                    apply_review_result(record, result, blacklist, ai_threshold, usd_to_thb)
                    set_flash("success" if result.get("ok") else "error", "AI Review สำเร็จ" if result.get("ok") else record["error"])
                    st.rerun()

            st.divider()
            render_api_cost(record, usd_to_thb)
            st.divider()
            st.subheader("AI Review")
            render_ai_review(record.get("ai_review"))
            if record.get("review_model_used"):
                st.caption(f"โมเดลที่ใช้ AI Review จริง: {record['review_model_used']}")

            if record.get("error"):
                st.error(record["error"])

            if show_debug and (record.get("diagnostics") or record.get("review_diagnostics")):
                with st.expander("Technical diagnostics / Raw response"):
                    st.json(
                        {
                            "analysis": record.get("diagnostics", []),
                            "review": record.get("review_diagnostics", []),
                        },
                        expanded=False,
                    )


init_state()
show_flash()

st.title("🧠 AI Stock Vision V2.5 — Cloud Safe")
st.caption(
    "สร้าง Adobe Stock Title + 49 Keywords พร้อมตรวจคุณภาพ ติดตาม Token/ค่า API ต่อภาพ และ Export CSV/ZIP"
)

# Cloud diagnostics: useful when Streamlit Community Cloud rebuilds with a new runtime.
import platform
st.caption(
    f"Runtime: Python {platform.python_version()} · Streamlit {st.__version__} · "
    f"App {APP_NAME}"
)

with st.sidebar:
    st.header("OpenAI Settings")
    manual_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="ไม่ถูกบันทึกลง CSV, Cache หรือ Log",
    )
    api_key, api_key_source = resolve_api_key(manual_api_key)
    st.caption(f"API Key source: {api_key_source}")

    if st.button("รีเฟรชรายการโมเดล", use_container_width=True):
        if not api_key:
            st.error("กรุณาใส่ API Key ก่อน")
        else:
            try:
                with st.spinner("กำลังอ่านรายชื่อโมเดลที่ API Key เข้าถึงได้..."):
                    models = discover_models(api_key)
                st.session_state.available_models = models
                st.session_state.selected_model_id = AUTO_MODEL_OPTION
                st.success(f"พบโมเดลที่มีแนวโน้มรองรับงานนี้ {len(models)} โมเดล")
            except Exception as exc:
                error = classify_api_error(exc, api_key)
                st.error(f"{error['friendly']}\n\n{error['detail']}")

    models = st.session_state.available_models
    model_options = [AUTO_MODEL_OPTION] + models
    if st.session_state.selected_model_id not in model_options:
        st.session_state.selected_model_id = AUTO_MODEL_OPTION
    selected_index = model_options.index(st.session_state.selected_model_id)
    selected_model = st.selectbox(
        "Model",
        model_options,
        index=selected_index,
        format_func=lambda value: (
            "Auto — เลือกโมเดลคุณภาพสูงสุดที่ใช้งานได้"
            if value == AUTO_MODEL_OPTION
            else value
        ),
        help="โหมด Auto จะใช้รายการจาก client.models.list() และสลับไปโมเดลสำรองเมื่อโมเดลแรกใช้ไม่ได้",
    )
    st.session_state.selected_model_id = selected_model
    custom_model = st.text_input(
        "Custom Model ID",
        placeholder="เช่น gpt-5.6-terra หรือ gpt-4.1 — ระบบจะลองแล้ว fallback ให้อัตโนมัติ",
    )
    auto_model_fallback = st.checkbox(
        "สลับไปโมเดลสำรองอัตโนมัติ",
        value=True,
        help="หากโมเดลไม่พบ ไม่รองรับภาพ ไม่รองรับ Structured Output หรือให้ผลลัพธ์ใช้ไม่ได้ ระบบจะลองตัวถัดไป",
    )

    requested_model = custom_model.strip() or selected_model
    effective_models = build_model_fallback_chain(
        models,
        selected_model=requested_model,
        auto_fallback=auto_model_fallback,
    )
    if effective_models:
        st.caption(f"โมเดลหลัก: {effective_models[0]}")
        if len(effective_models) > 1:
            preview = " → ".join(effective_models[:6])
            suffix = " → …" if len(effective_models) > 6 else ""
            st.caption(f"Fallback chain: {preview}{suffix}")

    if not models:
        st.warning(
            f"ยังไม่ได้ดึงรายการโมเดล ระบบจะลอง {DEFAULT_MODEL_ID} ก่อน "
            "แล้วถอยไป gpt-4.1 และ gpt-4o อัตโนมัติ"
        )

    st.divider()
    st.header("Processing")
    usd_to_thb = st.number_input(
        "อัตรา USD → THB สำหรับประมาณการ",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.usd_to_thb),
        step=0.10,
        help="ค่า API จริงคำนวณเป็น USD; ค่าเงินบาทเป็นเพียงประมาณการ",
    )
    st.session_state.usd_to_thb = usd_to_thb
    max_workers = st.slider("จำนวนภาพที่ทำพร้อมกัน", 1, 4, 2)
    use_cache = st.checkbox("ใช้ผล Cache เมื่อข้อมูลตรงกัน", value=True)
    auto_review = st.checkbox("ทำ AI Review อัตโนมัติหลังวิเคราะห์", value=False)
    only_pending = st.checkbox("วิเคราะห์เฉพาะภาพที่ยังไม่มีผล", value=True)
    ai_threshold = st.slider(
        "เกณฑ์ AI Review สำหรับ Export",
        0,
        100,
        DEFAULT_AI_REVIEW_THRESHOLD,
        5,
    )
    block_failed_export = st.checkbox("ไม่ให้ Export ภาพที่ไม่ผ่านคุณภาพ", value=True)
    show_debug = st.checkbox("แสดง Technical diagnostics", value=False)

    st.divider()
    st.header("Blacklist")
    blacklist_text = st.text_area(
        "หนึ่งคำต่อบรรทัด หรือคั่นด้วย comma",
        value="\n".join(sorted(DEFAULT_BLACKLIST)),
        height=220,
    )
    blacklist = parse_blacklist(blacklist_text)

st.subheader("1. อัปโหลดภาพ")
uploaded_files = st.file_uploader(
    "รองรับ JPG, JPEG และ PNG — เลือกได้หลายภาพ",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_nonce}",
)
sync_uploaded_files(uploaded_files or [])

reset_col, cache_col, cache_confirm_col = st.columns([1, 1.25, 1.2])
with reset_col:
    if st.button("ล้างภาพออกจากหน้าอัปโหลด", use_container_width=True):
        st.session_state.assets = {}
        st.session_state.asset_order = []
        st.session_state.export_bundle = None
        st.session_state.excluded_exports = []
        st.session_state.uploader_nonce += 1
        set_flash("success", "ล้างภาพและข้อมูลใน Session ปัจจุบันแล้ว")
        st.rerun()

with cache_confirm_col:
    confirm_clear_cache = st.checkbox("ยืนยันล้าง Server Cache", value=False)
with cache_col:
    if st.button(
        "ล้างแคชภาพทั้งหมดบนเซิร์ฟเวอร์",
        disabled=not confirm_clear_cache,
        use_container_width=True,
    ):
        try:
            file_count, bytes_removed = clear_app_cache()
            st.session_state.export_bundle = None
            set_flash("success", f"ลบ {file_count} ไฟล์ คืนพื้นที่ {human_bytes(bytes_removed)}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

assets = st.session_state.assets
asset_order = st.session_state.asset_order

if not asset_order:
    st.info("อัปโหลดภาพเพื่อเริ่มใช้งาน")
    st.stop()

st.subheader("2. วิเคราะห์ภาพ")
summary_cols = st.columns(4)
summary_cols[0].metric("จำนวนภาพ", len(asset_order))
summary_cols[1].metric(
    "วิเคราะห์แล้ว",
    sum(assets[asset_id]["analysis_status"] == "วิเคราะห์แล้ว" for asset_id in asset_order),
)
summary_cols[2].metric(
    "พร้อม Export",
    sum(
        bool((assets[asset_id].get("quality") or {}).get("export_eligible"))
        for asset_id in asset_order
    ),
)
summary_cols[3].metric(
    "ขนาดรวม",
    human_bytes(sum(assets[asset_id]["size_bytes"] for asset_id in asset_order)),
)

all_session_events: list[dict[str, Any]] = []
for asset_id in asset_order:
    all_session_events.extend(assets[asset_id].get("api_usage_events", []))
session_usage = summarize_usage_events(all_session_events, usd_to_thb)
usage_cols = st.columns(4)
usage_cols[0].metric("API Calls รวม", session_usage["api_calls"])
usage_cols[1].metric("Input Tokens รวม", f"{session_usage['input_tokens']:,}")
usage_cols[2].metric("Output Tokens รวม", f"{session_usage['output_tokens']:,}")
usage_cols[3].metric(
    "ค่า API รวม Session",
    format_usd(session_usage["known_cost_usd"]),
    f"≈ ฿{session_usage['estimated_cost_thb']:.4f}",
)
if session_usage["unknown_pricing_models"]:
    st.warning(
        "ยอดรวมยังไม่รวมโมเดลที่ไม่มีราคาในตาราง: "
        + ", ".join(session_usage["unknown_pricing_models"])
    )

batch_col, export_col = st.columns([1.2, 1])
with batch_col:
    if st.button("วิเคราะห์ภาพทั้งหมด", type="primary", use_container_width=True):
        if not api_key:
            st.error("กรุณาใส่ OpenAI API Key ก่อนวิเคราะห์")
        else:
            target_ids = [
                asset_id
                for asset_id in asset_order
                if not only_pending or assets[asset_id]["analysis_status"] != "วิเคราะห์แล้ว"
            ]
            target_ids = [
                asset_id
                for asset_id in target_ids
                if assets[asset_id]["image_info"].get("width", 0) > 0
            ]
            if not target_ids:
                st.info("ไม่มีภาพที่ต้องวิเคราะห์ตามเงื่อนไขปัจจุบัน")
            else:
                run_batch_analysis(
                    target_ids,
                    api_key=api_key,
                    model=effective_models,
                    blacklist=blacklist,
                    use_cache=use_cache,
                    auto_review=auto_review,
                    max_workers=max_workers,
                    ai_threshold=ai_threshold,
                    usd_to_thb=usd_to_thb,
                )

with export_col:
    if st.button("เตรียมไฟล์ CSV และ ZIP", use_container_width=True):
        selected_records: list[dict[str, Any]] = []
        excluded: list[str] = []
        for asset_id in asset_order:
            record = assets[asset_id]
            recompute_quality(record, blacklist, ai_threshold)
            record["api_usage_summary"] = summarize_usage_events(
                record.get("api_usage_events", []), usd_to_thb
            )
            if block_failed_export and not record["quality"]["export_eligible"]:
                excluded.append(record["filename"])
            else:
                selected_records.append(record)

        st.session_state.excluded_exports = excluded
        if not selected_records:
            st.error("ไม่มีไฟล์ที่ผ่านเงื่อนไขสำหรับ Export")
        else:
            try:
                with st.spinner("กำลังฝัง Metadata และสร้างไฟล์ ZIP..."):
                    st.session_state.export_bundle = build_export_zip(selected_records)
                st.success(f"เตรียม Export แล้ว {len(selected_records)} ภาพ")
            except Exception as exc:
                st.error(f"สร้างไฟล์ Export ไม่สำเร็จ: {type(exc).__name__}: {exc}")

if st.session_state.excluded_exports:
    st.warning("ไฟล์ที่ถูกกันออก: " + ", ".join(st.session_state.excluded_exports))

bundle = st.session_state.export_bundle
if bundle:
    download_cols = st.columns(4)
    download_cols[0].download_button(
        "ดาวน์โหลด Adobe CSV",
        data=bundle["adobe_csv"],
        file_name="adobe_stock_metadata.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[1].download_button(
        "ดาวน์โหลด Quality Report CSV",
        data=bundle["quality_csv"],
        file_name="quality_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[2].download_button(
        "ดาวน์โหลด API Cost CSV",
        data=bundle["api_cost_csv"],
        file_name="api_cost_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
    download_cols[3].download_button(
        "ดาวน์โหลด ZIP พร้อมภาพ",
        data=bundle["zip_bytes"],
        file_name=bundle["filename"],
        mime="application/zip",
        use_container_width=True,
    )
    if bundle.get("warnings"):
        with st.expander("Export warnings"):
            for warning in bundle["warnings"]:
                st.warning(warning)

st.divider()
st.subheader("3. ตรวจและแก้รายภาพ")
for current_asset_id in asset_order:
    render_asset(
        assets[current_asset_id],
        api_key=api_key,
        model=effective_models,
        blacklist=blacklist,
        use_cache=use_cache,
        ai_threshold=ai_threshold,
        show_debug=show_debug,
        usd_to_thb=usd_to_thb,
    )
