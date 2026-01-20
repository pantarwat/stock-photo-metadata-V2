import streamlit as st
import pandas as pd
from PIL import Image
import io
import zipfile
import openai
import base64
import re
import os
import tempfile
import traceback # เอาไว้จับ Error แบบละเอียด
from iptcinfo3 import IPTCInfo

# --- 1. Config ---
st.set_page_config(page_title="AI Stock Vision - Anti Crash", layout="wide")
st.title("🛡️ AI Stock Vision (Stable Version)")

# --- 2. Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # ปุ่มล้างค่า (Reset)
    if st.button("🗑️ ล้างข้อมูล/เริ่มใหม่ (กดเมื่อค้าง)", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
        
    api_key = st.text_input("🔑 API Key", type="password")
    model_choice = st.selectbox("🤖 Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    
    category_dict = {
        "1-Animals": 1, "2-Architecture": 2, "3-Business": 3, "4-Drinks": 4, 
        "5-Nature": 5, "6-Emotions": 6, "7-Food": 7, "8-Graphic": 8, 
        "11-Landscape": 11, "13-People": 13, "19-Technology": 19, "21-Travel": 21
    }
    selected_cat_name = st.selectbox("📁 Category", list(category_dict.keys()), index=4)
    user_hint = st.text_area("💡 Context Hint", placeholder="Ex: Sunset at beach")
    blacklist = [x.strip().lower() for x in st.text_area("🛡️ Blacklist", "nike, apple, logo").split(",")]

# --- 3. Functions ---
def analyze_image_safe(image_bytes, category, hint, key, model):
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        client = openai.OpenAI(api_key=key)
        
        prompt = (
            f"Analyze for Adobe Stock. Category: {category}. Context: {hint}\n"
            f"REQUIRED OUTPUT:\n"
            f"Title: [Start with top 7 keywords, 100-200 chars]\n"
            f"Keywords: [Exactly 49 keywords, comma separated]\n"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=500
        )
        raw_text = response.choices[0].message.content
        
        if "sorry" in raw_text.lower() and "assist" in raw_text.lower():
             return "Safety Filter Triggered", "Safety Filter Triggered", raw_text, True

        title_match = re.search(r"Title:\s*(.*?)(?=\n|Keywords:)", raw_text, re.IGNORECASE | re.DOTALL)
        keywords_match = re.search(r"Keywords:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
        
        t = title_match.group(1).strip() if title_match else ""
        k = keywords_match.group(1).strip() if keywords_match else ""
        
        if not t or not k: return raw_text, raw_text, raw_text, True
            
        return t, k, raw_text, False

    except Exception as e:
        return "Error", str(e), str(e), True

def process_to_jpg_iptc(uploaded_file_bytes, title, keywords):
    tmp_path = None
    try:
        img = Image.open(io.BytesIO(uploaded_file_bytes))
        if img.mode in ("RGBA", "P"): img = img.convert("RGB")
        
        # ใช้ suffix .jpg ชัดเจน
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, quality=100, subsampling=0)
            tmp_path = tmp.name
        
        # ใส่ try-except ซ้อนอีกชั้นกัน iptcinfo3 พัง
        try:
            info = IPTCInfo(tmp_path, force=True)
            info['object name'] = title.encode('utf-8')
            info['caption/abstract'] = title.encode('utf-8')
            keyword_list = [k.strip().encode('utf-8') for k in keywords.split(',') if k.strip()]
            info['keywords'] = keyword_list
            info.save()
        except Exception as iptc_err:
            print(f"IPTC Write Error (skipped): {iptc_err}")
            # ถ้าฝังไม่เข้าจริงๆ ให้ข้ามไป (ยังได้ไฟล์ภาพคืนมา)
        
        with open(tmp_path, 'rb') as f:
            final_bytes = f.read()
            
        return final_bytes
        
    except Exception as e:
        print(f"Critical Image Error: {e}")
        return uploaded_file_bytes 
    finally:
        # ลบไฟล์ขยะเสมอ ไม่ว่าจะ error หรือไม่
        if tmp_path:
            try:
                os.unlink(tmp_path)
                if os.path.exists(tmp_path + "~"): os.unlink(tmp_path + "~")
            except: pass

# --- 4. Main UI with Global Try-Except ---
try:
    uploaded_images = st.file_uploader("📸 อัปโหลดรูปภาพ (แนะนำทีละไม่เกิน 20 รูป)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if "results" not in st.session_state:
        st.session_state.results = {}

    if uploaded_images:
        # ปุ่มเริ่ม
        if st.button("🚀 เริ่มวิเคราะห์", use_container_width=True, type="primary"):
            if not api_key:
                st.error("❌ ลืมใส่ API Key")
            else:
                bar = st.progress(0)
                status = st.empty()
                
                for i, img_file in enumerate(uploaded_images):
                    # เช็คว่าทำไปแล้วหรือยัง (ประหยัดเงิน + ประหยัดเมม)
                    if img_file.name not in st.session_state.results:
                        status.text(f"Processing: {img_file.name}...")
                        
                        # อ่านไฟล์แค่ครั้งเดียว
                        file_bytes = img_file.read()
                        t, k, raw, err = analyze_image_safe(file_bytes, selected_cat_name, user_hint, api_key, model_choice)
                        
                        # เก็บลง session
                        st.session_state.results[img_file.name] = {
                            "t": t, "k": k, "raw": raw, "err": err, 
                            "bytes": file_bytes # จำเป็นต้องเก็บไว้เพื่อสร้าง ZIP
                        }
                    
                    bar.progress((i + 1) / len(uploaded_images))
                
                status.success("✅ เสร็จสิ้น!")

        # แสดงผล
        if st.session_state.results:
            st.divider()
            final_data = []
            
            # วนลูปจาก session_state แทน เพื่อกันไฟล์หาย
            keys_to_remove = []
            
            for filename, data in st.session_state.results.items():
                # กรองเฉพาะไฟล์ที่อยู่ใน upload ปัจจุบัน (เผื่อ User ลบไฟล์ออก)
                if not any(u.name == filename for u in uploaded_images):
                    continue

                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    c1.image(data['bytes'], width=150)
                    
                    if data['err']: st.warning("⚠️ ตรวจสอบข้อมูล")
                        
                    et = c2.text_input("Title", value=data['t'], key=f"t_{filename}")
                    ek = c2.text_area("Keywords", value=data['k'], key=f"k_{filename}")
                    
                    raw_k = [x.strip() for x in ek.split(',') if x.strip()]
                    clean_k = [x for x in raw_k if x.lower() not in blacklist][:49]
                    final_k_str = ", ".join(clean_k)
                    
                    c2.caption(f"Keywords: {len(clean_k)}/49")
                    
                    final_data.append({
                        "Filename": filename.split('.')[0] + ".jpg",
                        "Title": et,
                        "Keywords": final_k_str,
                        "Category": category_dict[selected_cat_name],
                        "Releases": "",
                        "original_bytes": data['bytes']
                    })

            if final_data:
                st.write(f"จำนวนภาพพร้อมโหลด: {len(final_data)} ภาพ")
                d1, d2 = st.columns(2)
                
                # CSV
                df = pd.DataFrame(final_data)[["Filename", "Title", "Keywords", "Category", "Releases"]]
                d1.download_button("📊 Download CSV", df.to_csv(index=False).encode('utf-8'), "adobe_stock.csv", use_container_width=True)
                
                # ZIP (Process on click to save memory)
                if d2.button("📥 เตรียมไฟล์ ZIP (คลิกแล้วรอสักครู่)", use_container_width=True):
                    with st.spinner("กำลังฝัง Metadata... (ขั้นตอนนี้ใช้ CPU สูง)"):
                        z_buf = io.BytesIO()
                        with zipfile.ZipFile(z_buf, "a", zipfile.ZIP_DEFLATED) as zf:
                            for item in final_data:
                                final_jpg = process_to_jpg_iptc(item['original_bytes'], item['Title'], item['Keywords'])
                                zf.writestr(item['Filename'], final_jpg)
                        
                        st.download_button("📂 คลิกเพื่อโหลด ZIP", z_buf.getvalue(), "images_iptc.zip", use_container_width=True)

except Exception as e:
    # นี่คือตัวจับ Error ระดับชาติ ไม่ให้จอขาว
    st.error("🚨 เกิดข้อผิดพลาดร้ายแรง (Application Error)")
    st.code(traceback.format_exc()) # โชว์สาเหตุให้เห็นชัดๆ
    st.info("คำแนะนำ: กดปุ่ม 'ล้างข้อมูล/เริ่มใหม่' ในแถบซ้ายมือ หรือกด F5 เพื่อรีเฟรชหน้าเว็บ")