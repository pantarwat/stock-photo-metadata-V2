import streamlit as st
import pandas as pd
from PIL import Image
import io
import zipfile
import base64
import re
import os
import tempfile
import traceback
from iptcinfo3 import IPTCInfo
from openai import OpenAI

# =========================
# 1) PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Stock Vision - GPT-5 Ready", layout="wide")
st.title("🎯 AI Stock Vision (GPT-5 Ready)")
st.caption("สร้าง Title / Keywords / CSV / ZIP พร้อมฝัง IPTC Metadata สำหรับงาน Stock")

# =========================
# 2) HELPERS
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

def sanitize_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    name, _ = os.path.splitext(filename)
    name = re.sub(r"[^\w\-. ]+", "_", name).strip()
    return (name or "image") + ".jpg"

def normalize_keywords(raw_keywords: str, blacklist_words: list[str]) -> str:
    blacklist_set = {x.strip().lower() for x in blacklist_words if x.strip()}
    parts = [x.strip() for x in raw_keywords.split(",") if x.strip()]

    cleaned = []
    seen = set()

    for kw in parts:
        kw_norm = re.sub(r"\s+", " ", kw).strip()
        kw_key = kw_norm.lower()
        if not kw_norm:
            continue
        if kw_key in blacklist_set:
            continue
        if kw_key in seen:
            continue
        seen.add(kw_key)
        cleaned.append(kw_norm)
        if len(cleaned) >= 49:
            break

    return ", ".join(cleaned)

def parse_model_output(raw_text: str):
    title_match = re.search(r"Title:\s*(.*?)(?=\n\s*Keywords:|\Z)", raw_text, re.IGNORECASE | re.DOTALL)
    keywords_match = re.search(r"Keywords:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)

    title = title_match.group(1).strip() if title_match else ""
    keywords = keywords_match.group(1).strip() if keywords_match else ""

    if keywords:
        keywords = keywords.replace("\n", " ").strip()
        keywords = re.sub(r"^\[|\]$", "", keywords).strip()

    return title, keywords

def analyze_image_with_openai(
    image_bytes: bytes,
    category_name: str,
    category_num: int,
    hint: str,
    api_key: str,
    model: str,
):
    """
    ใช้ OpenAI Responses API ซึ่งเป็น API รุ่นใหม่สำหรับ multimodal input
    และเหมาะกับโมเดลรุ่นล่าสุด รวมถึง GPT-5 family
    """
    try:
        client = OpenAI(api_key=api_key)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = f"""
You are a professional Adobe Stock contributor and metadata editor.

Analyze the uploaded image and produce Adobe Stock-ready metadata.

Context hint from user: {hint if hint.strip() else "None"}
Adobe category chosen by user: {category_name} (Category ID: {category_num})

Your tasks:

1) TITLE
- Write exactly 1 English title.
- Length should be about 100 to 200 characters.
- Make it natural, descriptive, fluent, and commercially useful.
- Do NOT write a keyword dump or comma list.
- Do NOT mention brands, trademarks, logos, copyrighted characters, or restricted brand names.
- Prefer clear stock-friendly language.

2) KEYWORDS
- Generate exactly 49 English keywords.
- Rank them by importance, most important first.
- Separate keywords with commas only.
- Use singular/plural only when truly useful; avoid spammy duplication.
- Avoid brands, logos, trademarks, or copyrighted names.
- Avoid irrelevant filler keywords.

Output format must be EXACTLY:

Title: <one sentence title>
Keywords: keyword1, keyword2, keyword3, ...

Do not add any explanation before or after.
""".strip()

        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    ],
                }
            ],
            max_output_tokens=700,
        )

        raw_text = (response.output_text or "").strip()

        if not raw_text:
            return {
                "title": "Error",
                "keywords": "",
                "raw": "Model returned empty output.",
                "error": True,
            }

        title, keywords = parse_model_output(raw_text)

        if not title or not keywords:
            return {
                "title": "Parse Error",
                "keywords": "",
                "raw": raw_text,
                "error": True,
            }

        return {
            "title": title,
            "keywords": keywords,
            "raw": raw_text,
            "error": False,
        }

    except Exception as e:
        return {
            "title": "Error",
            "keywords": "",
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
            # ถ้า IPTC ฝังไม่ได้ ก็ยังคืน JPG ปกติ
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

# =========================
# 3) SESSION STATE
# =========================
if "results" not in st.session_state:
    st.session_state.results = {}

if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []

# =========================
# 4) SIDEBAR
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
        placeholder="เช่น: chess piece with stock graph, modern business concept, startup growth"
    )

    blacklist_raw = st.text_area(
        "🛡️ Blacklist Keywords",
        value="nike, apple, logo, brand, trademark"
    )
    blacklist = [x.strip().lower() for x in blacklist_raw.split(",") if x.strip()]

    st.divider()
    st.markdown("**แพ็กเกจที่ควรติดตั้ง**")
    st.code("pip install streamlit pandas pillow openai iptcinfo3")

# =========================
# 5) MAIN UI
# =========================
try:
    uploaded_images = st.file_uploader(
        "📸 อัปโหลดรูปภาพ",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:
        current_names = [u.name for u in uploaded_images]

        # ล้างผลลัพธ์เฉพาะไฟล์ที่ไม่อยู่แล้ว
        for old_name in list(st.session_state.results.keys()):
            if old_name not in current_names:
                del st.session_state.results[old_name]

        st.session_state.last_uploaded_names = current_names

        col_a, col_b = st.columns([1, 1])

        with col_a:
            if st.button("🚀 เริ่มวิเคราะห์", use_container_width=True, type="primary"):
                if not api_key.strip():
                    st.error("❌ กรุณาใส่ OpenAI API Key ก่อน")
                elif not model_choice.strip():
                    st.error("❌ กรุณาเลือกหรือกรอกชื่อโมเดล")
                else:
                    progress = st.progress(0)
                    status = st.empty()

                    for i, img_file in enumerate(uploaded_images):
                        status.info(f"กำลังวิเคราะห์: {img_file.name}")

                        file_bytes = img_file.read()
                        result = analyze_image_with_openai(
                            image_bytes=file_bytes,
                            category_name=selected_cat_name,
                            category_num=CATEGORY_DICT[selected_cat_name],
                            hint=user_hint,
                            api_key=api_key,
                            model=model_choice,
                        )

                        st.session_state.results[img_file.name] = {
                            "title": result["title"],
                            "keywords": result["keywords"],
                            "raw": result["raw"],
                            "error": result["error"],
                            "bytes": file_bytes,
                        }

                        progress.progress((i + 1) / len(uploaded_images))

                    status.success("✅ วิเคราะห์เสร็จแล้ว")

        with col_b:
            if st.button("♻️ วิเคราะห์ใหม่ทั้งหมด", use_container_width=True):
                for img_file in uploaded_images:
                    if img_file.name in st.session_state.results:
                        del st.session_state.results[img_file.name]
                st.rerun()

    # =========================
    # 6) RESULTS
    # =========================
    if uploaded_images and st.session_state.results:
        st.divider()
        st.subheader("ผลลัพธ์")

        final_data = []

        for img_file in uploaded_images:
            filename = img_file.name
            if filename not in st.session_state.results:
                continue

            data = st.session_state.results[filename]

            with st.container(border=True):
                c1, c2 = st.columns([1, 2])

                with c1:
                    st.image(data["bytes"], use_container_width=True)
                    st.caption(filename)

                with c2:
                    if data["error"]:
                        st.error("มีปัญหาในการวิเคราะห์ไฟล์นี้")
                        with st.expander("ดูข้อความ error / raw output"):
                            st.code(data["raw"])

                    edited_title = st.text_input(
                        "Title",
                        value=data["title"],
                        key=f"title_{filename}",
                    )

                    edited_keywords = st.text_area(
                        "Keywords",
                        value=data["keywords"],
                        key=f"keywords_{filename}",
                        height=130,
                    )

                    cleaned_keywords = normalize_keywords(edited_keywords, blacklist)
                    cleaned_count = len([x for x in cleaned_keywords.split(",") if x.strip()])

                    st.caption(f"Keyword count หลัง clean: {cleaned_count}/49")

                    if cleaned_count < 49:
                        st.warning("คีย์เวิร์ดยังไม่ถึง 49 คำ")
                    elif cleaned_count > 49:
                        st.warning("คีย์เวิร์ดเกิน 49 คำ ระบบจะตัดให้เหลือ 49")
                    else:
                        st.success("คีย์เวิร์ดครบ 49 คำแล้ว")

                    safe_name = sanitize_filename(filename)

                    final_data.append({
                        "Filename": safe_name,
                        "Title": edited_title.strip(),
                        "Keywords": cleaned_keywords,
                        "Category": CATEGORY_DICT[selected_cat_name],
                        "Releases": "",
                        "original_bytes": data["bytes"],
                    })

        # =========================
        # 7) EXPORT
        # =========================
        if final_data:
            st.divider()
            st.subheader("Export")

            export_df = pd.DataFrame(final_data)[["Filename", "Title", "Keywords", "Category", "Releases"]]

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
                            # ใส่ csv ลงไปด้วย
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