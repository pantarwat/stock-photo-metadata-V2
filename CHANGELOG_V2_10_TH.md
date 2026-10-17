# AI Stock Vision V2.10 — Media Safe

แก้ปัญหา Preview ภาพขึ้นเป็นรูปแตก และกด Download แล้ว browser แจ้งว่า site wasn't available / 404 บน Streamlit Community Cloud

- อัป Streamlit จาก 1.59.2 เป็น 1.65.0 ซึ่งรวม media/download/range-response fix จาก 1.62+
- Preview ใช้ inline data URI จึงไม่พึ่ง Streamlit `/media/...`
- ZIP สร้างด้วย `BytesIO` ใน memory และเก็บ `zip_bytes` ใน Session
- ปุ่ม ZIP ส่ง bytes โดยตรง ไม่อ่าน temp file ตอนกด
- CSV ยังเป็น bytes ใน memory
- Metadata embedding + verification และ session-safe upload เดิมยังอยู่ครบ
- ถ้า ZIP เกิน 180 MB จะแจ้งเตือนให้แบ่ง Export

Deploy: ต้องทับทั้ง `app.py` และ `requirements.txt` แล้ว Reboot app เพื่อให้ Cloud ติดตั้ง Streamlit 1.65.0 ใหม่
