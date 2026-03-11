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
import traceback
from iptcinfo3 import IPTCInfo

# --- 1. Config ---
st.set_page_config(page_title="AI Stock Vision - Expert Mode", layout="wide")
st.title("🎯 AI Stock Vision (GPT-5.4 Expert Mode)")

# --- 2. Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🗑️ ล้างข้อมูล/เริ่มใหม่", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
        
    api_key = st.text_input("🔑 API Key", type="password")
    
    # --- อัปเดตเมนู Model ใหม่ ---
    model_options = {
        "GPT-5.4 (ผู้เชี่ยวชาญ)": "gpt-5.4", # เปลี่ยนรหัส API ตรงนี้ได้ถ้า OpenAI ใช้ชื่ออื่น
        "GPT-4o (ตัวท็อปมาตรฐาน)": "gpt-4o",
        "GPT-4o-mini (ประหยัดงบ)": "gpt-4o-mini"
    }
    selected_model_display = st.selectbox("🤖 Model", list(model_options.keys()), index=0)
    model_choice = model_options[selected_model_display]
    
    category_dict = {
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
    
    selected_cat_name = st.selectbox("📁 Adobe Category", list(category_dict.keys()), index=2)
    
    st.divider()
    user_hint = st.text_area("💡 Context Hint", placeholder="Ex: Chess piece with stock graph")
    blacklist = [x.strip().lower() for x in st.text_area("🛡️ Blacklist", "nike, apple, logo").split(",")]

# --- 3. AI Function ---
def analyze_image_sentence(image_bytes, category, hint, key, model):
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        client = openai.OpenAI(api_key=key)
        
        prompt = (
            f"Act as a professional Adobe Stock contributor. Analyze this image. Category: {category}. Context: {hint}\n\n"
            f"TASK 1: KEYWORDS\n"
            f"Generate exactly 49 keywords relevant to the image, ranked by importance.\n\n"
            f"TASK 2: TITLE (CRITICAL STEP)\n"
            f"Write a single descriptive sentence (100-200 characters) that naturally INCORPORATES the top 7 most important keywords.\n"
            f"- BAD Title: Chess piece, stock market, business strategy, finance growth.\n"
            f"- GOOD Title: A chess piece standing before a glowing stock market graph, illustrating business strategy, finance growth, and investment success.\n"
            f"Rules: Do not use comma lists. Make it a fluid, grammatical sentence.\n\n"
            f"REQUIRED OUTPUT FORMAT:\n"
            f"Title: [Your descriptive sentence]\n"
            f"Keywords: [word1, word2, ...]\n"
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
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name, quality=100, subsampling=0)
            tmp_path = tmp.name
        
        try:
            info = IPTCInfo(tmp_path, force=True)
            info['object name'] = title.encode('utf-8')
            info['caption/abstract'] = title.encode('utf-8')
            keyword_list = [k.strip().encode('utf-8') for k in keywords.split(',') if k.strip()]
            info['keywords'] = keyword_list
            info.save()
        except Exception as iptc_err:
            pass
        
        with open(tmp_path, 'rb') as f:
            final_bytes = f.read()
            
        return final_bytes
        
    except Exception as e:
        return uploaded_file_bytes 
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
                if os.path.exists(tmp_path + "~"): os.unlink(tmp_path + "~")
            except: pass

# --- 4. Main UI ---
try:
    uploaded_images = st.file_uploader("📸 อัปโหลดรูปภาพ", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if "results" not in st.session_state:
        st.session_state.results = {}

    if uploaded_images:
        if st.button("🚀 เริ่มวิเคราะห์ (Expert Model)", use_container_width=True, type="primary"):
            if not api_key:
                st.error("❌ ลืมใส่ API Key")
            else:
                bar = st.progress(0)
                
                for i, img_file in enumerate(uploaded_images):
                    if img_file.name not in st.session_state.results:
                        file_bytes = img_file.read()
                        t, k, raw, err = analyze_image_sentence(file_bytes, selected_cat_name, user_hint, api_key, model_choice)
                        st.session_state.results[img_file.name] = {"t": t, "k": k, "raw": raw, "err": err, "bytes": file_bytes}
                    bar.progress((i + 1) / len(uploaded_images))
                st.success("✅ เรียบร้อย!")

        if st.session_state.results:
            st.divider()
            final_data = []
            
            for filename, data in st.session_state.results.items():
                if not any(u.name == filename for u in uploaded_images): continue

                with st.container(border=True):
                    c1, c2 = st.columns([1, 2])
                    c1.image(data['bytes'], width=150)
                    
                    et = c2.text_input("Title", value=data['t'], key=f"t_{filename}")
                    ek = c2.text_area("Keywords", value=data['k'], key=f"k_{filename}")
                    
                    raw_k = [x.strip() for x in ek.split(',') if x.strip()]
                    clean_k = [x for x in raw_k if x.lower() not in blacklist][:49]
                    final_k_str = ", ".join(clean_k)
                    
                    c2.caption(f"Count: {len(clean_k)}/49")
                    
                    final_data.append({
                        "Filename": filename.split('.')[0] + ".jpg",
                        "Title": et,
                        "Keywords": final_k_str,
                        "Category": category_dict[selected_cat_name],
                        "Releases": "",
                        "original_bytes": data['bytes']
                    })

            if final_data:
                d1, d2 = st.columns(2)
                df = pd.DataFrame(final_data)[["Filename", "Title", "Keywords", "Category", "Releases"]]
                d1.download_button("📊 Download CSV", df.to_csv(index=False).encode('utf-8'), "adobe_stock.csv", use_container_width=True)
                
                if d2.button("📥 เตรียมไฟล์ ZIP", use_container_width=True):
                    with st.spinner("กำลังฝังข้อมูล..."):
                        z_buf = io.BytesIO()
                        with zipfile.ZipFile(z_buf, "a", zipfile.ZIP_DEFLATED) as zf:
                            for item in final_data:
                                final_jpg = process_to_jpg_iptc(item['original_bytes'], item['Title'], item['Keywords'])
                                zf.writestr(item['Filename'], final_jpg)
                        st.download_button("📂 คลิกเพื่อโหลด ZIP", z_buf.getvalue(), "images_expert.zip", use_container_width=True)

except Exception as e:
    st.error("Application Error")
    st.code(traceback.format_exc())