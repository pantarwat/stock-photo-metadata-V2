# AI Stock Vision V2.9 — Session Safe

แก้ปัญหาหลักจาก V2.7 ที่พบในการใช้งานจริงบน Streamlit Community Cloud:

- ไม่ล้างรายการภาพเมื่อ `st.file_uploader` ส่งค่ากลับมาว่างชั่วคราวระหว่าง rerun/reconnect
- เก็บไฟล์ต้นฉบับลงโฟลเดอร์ temp ของแอป (`uploads/`) และอ่านคืนจาก disk cache ได้
- ลดการเก็บไฟล์ต้นฉบับขนาดใหญ่ไว้ใน `st.session_state` เพื่อลด memory/WebSocket pressure
- สร้าง preview ขนาดย่อแยก (`previews/`) แทนการส่งภาพ 4K/8K ต้นฉบับไปหน้า UI ทุก rerun
- Analysis / AI Review / Metadata Embed / ZIP ใช้ไฟล์ต้นฉบับจาก durable upload cache
- ZIP ไม่ถูกเก็บเป็น bytes ก้อนใหญ่ใน session state; เก็บ path + size และอ่านเมื่อคลิกดาวน์โหลด
- Download buttons ใช้ `on_click="ignore"` และ stable keys เพื่อไม่ให้การดาวน์โหลด trigger rerun
- ปุ่ม "ล้างภาพออกจากหน้าอัปโหลด" ยังคงเป็นวิธี explicit สำหรับล้าง asset/session
- ปุ่มล้าง Server Cache จะล้าง analysis/title/quality/export/upload/preview cache เฉพาะโฟลเดอร์ของแอป

ไม่เปลี่ยน logic หลักของ OpenAI, Quality Rule, Metadata embedding และ CSV schema จากสาย V2.7
