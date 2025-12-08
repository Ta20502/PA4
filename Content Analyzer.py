import streamlit as st
import pandas as pd
import json
import io
from google import genai
from google.genai.errors import APIError

# ==============================================================================
# 0. การตั้งค่าหน้าเว็บ (Page Configuration)
# ==============================================================================

st.set_page_config(
    page_title="📰 Content Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

def clear_all():
    st.session_state.analysis_result = None
    st.session_state.input_text = ""

# ==============================================================================
# 1. ฟังก์ชันหลักในการสร้าง System Prompt
# ==============================================================================

def create_system_prompt(n: int, summary_language: str) -> str:
    if summary_language == "Thai":
        summary_instruction = "**Write a CONCISE summary in THAI language.**"
        analysis_instruction = "The values for 'tone_analysis' and 'readability_level' MUST be translated into THAI with 1-2 sentences rationale."
        pos_instruction = "Translate 'part_of_speech' to THAI (e.g., Noun -> คำนาม, Verb -> คำกริยา)."
        tone_example = "เช่น: 'เป็นกลาง: บทความเน้นข้อเท็จจริง'"
        readability_example = "เช่น: 'ระดับมหาวิทยาลัย: มีศัพท์เฉพาะทางมาก'"
    else:
        summary_instruction = "**Write a CONCISE summary in ENGLISH language.**"
        analysis_instruction = "The values for 'tone_analysis' and 'readability_level' MUST be in ENGLISH with 1-2 sentences rationale."
        pos_instruction = "Keep 'part_of_speech' in ENGLISH (e.g., Noun, Verb)."
        tone_example = "e.g.: 'Neutral: Focuses on technical facts.'"
        readability_example = "e.g.: 'College Level: High specialized vocab.'"

    return f"""
You are an expert Content Analyzer. Return result strictly in JSON:
- 'analysis_summary': {{'summary_text', 'tone_analysis', 'readability_level'}}
- 'keyword_frequency': [{{'keyword', 'frequency_count', 'part_of_speech'}}] (top {n})

Rules:
1. {summary_instruction}
2. Tone: {tone_example}
3. Readability: {readability_example}
4. {analysis_instruction}
5. {pos_instruction}
"""

# ==============================================================================
# 2. ฟังก์ชันเรียกใช้ Gemini API
# ==============================================================================

def get_gemini_response(api_key: str, system_prompt: str, user_text: str) -> str | None:
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=[system_prompt, f"ARTICLE TEXT:\n\n{user_text}"],
            config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# ==============================================================================
# 3. ส่วนหลักของ Streamlit (Main App)
# ==============================================================================

# --- Initialization of Session State ---
# ส่วนสำคัญ: ใช้เก็บข้อมูลวิเคราะห์ไม่ให้หายเมื่อมีการ Rerun (เช่นตอนกด Download)
if 'article_input' not in st.session_state:
    st.session_state.article_input = ""

with st.sidebar:
    st.title("⚙️ การตั้งค่าระบบ")
    user_api_key = st.text_input("**API Key**", type="password")
    st.markdown("---")
    top_n_keywords = st.slider("เลือกจำนวนคำศัพท์สำคัญ", 10, 50, 10, 5)
    summary_language = st.selectbox("เลือกภาษาสำหรับวิเคราะห์", ["English", "Thai"])
    
    # ปุ่มสำหรับล้างผลลัพธ์
    if st.button("🗑️ ล้างข้อมูล"):
        st.session_state.article_input = ""
        st.rerun()

st.title('📰 Content Analyzer')
st.markdown('วิเคราะห์บทความเพื่อ**สรุป โทน และคำศัพท์สำคัญ**')

article_text = st.text_area(
    "ป้อนบทความที่ต้องการวิเคราะห์:",
    value=st.session_state.article_input,
    height=200,
    key="current_text"
)

st.session_state.article_input = article_text

# --- ปุ่มกดวิเคราะห์ ---
if st.button('🚀 วิเคราะห์เนื้อหา'):
    if not user_api_key:
        st.error("❌ โปรดใส่ API Key ใน Sidebar")
    elif not article_text.strip():
        st.error("❌ โปรดป้อนเนื้อหา")
    else:
        current_system_prompt = create_system_prompt(top_n_keywords, summary_language)
        with st.spinner("⏳ กำลังประมวลผล..."):
            raw_json = get_gemini_response(user_api_key, current_system_prompt, article_text)
            if raw_json:
                try:
                    # เก็บผลลัพธ์ลง Session State
                    st.session_state.analysis_result = json.loads(raw_json)
                except Exception as e:
                    st.error(f"JSON Parsing Error: {e}")

# ==============================================================================
# 4. ส่วนการแสดงผล (ดึงจาก Session State)
# ==============================================================================

# เช็คว่าใน Session State มีข้อมูลอยู่ไหม ถ้ามีให้วาด UI ออกมา
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    analysis_summary = result.get('analysis_summary', {})
    keyword_frequency = result.get('keyword_frequency', [])

    st.success("✅ วิเคราะห์สําเร็จ")

    # --- Section 1: สรุป ---
    st.header("1. สรุปและวิเคราะห์ภาพรวม")
    summary_df = pd.DataFrame({
        "เนื้อหา": [
            analysis_summary.get('summary_text', ''),
            analysis_summary.get('tone_analysis', ''),
            analysis_summary.get('readability_level', '')
        ]
    }, index=['⭐ Summary', '🗣️ Tone Analysis', '📚 Readability Level'])
    
    st.table(summary_df)

    # --- Section 2: คำศัพท์ ---
    st.header(f"2. คำศัพท์สำคัญ (Top {top_n_keywords})")
    st.markdown("จํานวนคําศัพท์ที่ต้องการจัดอันดับขึ้นอยู่กับบทความที่นํามาวิเคราะห์")
    freq_df = pd.DataFrame(keyword_frequency).head(top_n_keywords)
    # เปลี่ยนชื่อ Column ให้สวยงาม
    freq_df.columns = ['คำศัพท์ (Keyword)', 'ความถี่ (Count)', 'หน้าที่ (POS)']
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
    with col2:
        st.bar_chart(freq_df.set_index('คำศัพท์ (Keyword)')['ความถี่ (Count)'])

    # --- Section 3: ดาวน์โหลด ---
    st.markdown("---")
    st.header("3. ดาวน์โหลดข้อมูล")
    
    # ดาวน์โหลดสรุป
    summary_csv = summary_df.to_csv().encode('utf-8-sig')
    # ดาวน์โหลดคำศัพท์
    freq_csv = freq_df.to_csv(index=False).encode('utf-8-sig')

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download Summary (CSV)",
            summary_csv,
            "analysis_summary.csv",
            "text/csv",
            key="dl_sum"
        )
    with dl_col2:
        st.download_button(
            "⬇️ Download Keywords (CSV)",
            freq_csv,
            "keywords.csv",
            "text/csv",
            key="dl_freq"
        )
















