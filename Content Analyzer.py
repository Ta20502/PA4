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

# ==============================================================================
# 1. ฟังก์ชันหลักในการสร้าง System Prompt
# ==============================================================================

def create_system_prompt(n: int, summary_language: str) -> str:
    """สร้าง System Prompt สำหรับ LLM เพื่อให้ได้ผลลัพธ์ JSON ตามที่กำหนด"""
    if summary_language == "Thai":
        summary_instruction = "Summarize the article briefly in 2-3 sentences. **The summary must be in Thai.**"
    elif summary_language == "English":
        summary_instruction = "Summarize the article briefly in 2-3 sentences. **The summary must be in English.**"
    return f"""
You are an expert Content Analyzer and Linguist. Your task is to analyze the provided NEWS ARTICLE or TEXT.
You must perform four major tasks:
1.  **Summarize** the article briefly in 2-3 sentences.
2.  **Analyze the Tone (Sentiment)** of the article (e.g., Positive, Negative, Neutral, Informative).
3.  **Calculate the Frequency** of the {n} most important, non-stop-word nouns and verbs in the text.
4.  **Assess Readability** and suggest a reader level (e.g., High School, College, General Public).

Return the result *strictly* in a valid JSON object with the following three main keys:
-   'analysis_summary': A single object containing the summary and general analysis.
    -   'summary_text': The 2-3 sentence summary.
    -   'tone_analysis': The overall sentiment (e.g., Positive, Negative, Neutral).
    -   'readability_level': The suggested reader level.
-   'keyword_frequency': A JSON array of the {n} most important keywords.
    -   Each element in this array must be an object with keys: 'keyword', 'frequency_count', and 'part_of_speech'.

DO NOT include any introductory or concluding text outside the JSON object.
"""

# ==============================================================================
# 2. ฟังก์ชันเรียกใช้ Gemini API
# ==============================================================================

def get_gemini_response(api_key: str, system_prompt: str, user_text: str) -> str | None:
    """เรียกใช้ Google Gemini API เพื่อวิเคราะห์ข้อความและรับผลลัพธ์เป็น JSON"""
    if not api_key:
        st.error("❌ โปรดใส่ API Key ในช่อง 'API key' ด้านซ้ายก่อน")
        return None

    try:
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        
        response = client.models.generate_content(
            model=model,
            contents=[system_prompt, f"ARTICLE TEXT:\n\n{user_text}"],
            config={"response_mime_type": "application/json"} # สั่งให้ LLM สร้างผลลัพธ์เป็น JSON
        )
        
        return response.text

    except APIError as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการเรียกใช้ API: โปรดตรวจสอบ API Key ของคุณ ({e})")
        return None
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {e}")
        return None

# ==============================================================================
# 3. ส่วนหลักของ Streamlit (Main App)
# ==============================================================================

# --- 3.1 Sidebar สำหรับ API Key และการตั้งค่า ---
with st.sidebar:
    st.title("⚙️ การตั้งค่าระบบ")
    
    # API Key Input
    user_api_key = st.text_input(
        "**API Key**", 
        type="password",
        help="กรุณาใส่ API Key ของคุณ"
    )

    st.markdown("---")
    st.title("🔢 การตั้งค่าการวิเคราะห์คำศัพท์")
    
    # Slider สำหรับเลือก Top keyword
    top_n_keywords = st.slider(
        "เลือกอันดับคําศัพท์สําคัญที่ต้องการแสดงผล",
        min_value=10,
        max_value=50,
        value=10, # ค่าเริ่มต้น
        step=5,
    )
    st.info(f"แสดงผลลัพธ์คำศัพท์สำคัญ {top_n_keywords} อันดับ")

    st.markdown("---")
    st.title("🗣️ การตั้งค่าภาษา")
    
    # **🔥 NEW: Select box สำหรับเลือกภาษาที่ต้องการสรุป**
    summary_language = st.selectbox(
        "เลือกภาษาสำหรับ **บทสรุป**",
        options=["English", "Thai", "Original (ตามภาษาของบทความ)"],
        index=0, # English เป็นค่าเริ่มต้น
        help="ภาษาที่ใช้ในการสร้างข้อความสรุป (Summary Text) โดย AI"
    )
    # แปลงค่าที่เลือกจากภาษาไทยเป็นค่าที่จะใช้ใน prompt
    if "Thai" in summary_language:
        lang_code = "Thai"
    elif "English" in summary_language:
        lang_code = "English"

# --- 3.2 Main Content Area ---

st.title('📰 Content Analyzer: เครื่องมือวิเคราะห์เนื้อหาเชิงลึก')
st.markdown('ป้อน **บทความ ข่าว หรือข้อความ** ที่ต้องการวิเคราะห์เพื่อ **สรุป โทน และความถี่คำศัพท์สำคัญ**')

# Input Text Area
article_text = st.text_area(
    "ป้อนบทความ ข่าว หรือข้อความที่ต้องการวิเคราะห์ที่นี่:",
    value="Large language models (LLMs) are deep learning models trained on vast amounts of text data. They can understand, summarize, and generate human-like text, making them revolutionary tools for various NLP applications. The development of LLMs requires immense computational resources, particularly high-end GPUs. Despite their power, LLMs still face challenges related to factual accuracy and ethical bias.",
    height=250,
    placeholder="ตัวอย่าง: ป้อนบทความภาษาอังกฤษหรือภาษาไทยเพื่อเริ่มต้น"
)

# Submit Button
if st.button('🚀 วิเคราะห์เนื้อหา'):
    
    # 3.3 การตรวจสอบข้อมูลเริ่มต้น
    if not user_api_key:
        st.error("❌ โปรดใส่ **API Key** ในช่อง 'API key' ด้านซ้ายก่อน")
    elif not article_text or article_text.strip() == "":
        st.error("❌ โปรดป้อนข้อความบทความเพื่อทำการวิเคราะห์")
    else:
        # สร้าง SYSTEM_PROMPT ด้วยค่า N ที่ผู้ใช้เลือก
        current_system_prompt = create_system_prompt(top_n_keywords, lang_code)
        
        # 3.4 เรียกใช้ API และประมวลผล
        with st.spinner("⏳ กำลังวิเคราะห์เนื้อหาเชิงปริมาณ..."):
            json_response_text = get_gemini_response(
                api_key=user_api_key,
                system_prompt=current_system_prompt,
                user_text=article_text
            )

        if json_response_text:
            try:
                # แปลงผลลัพธ์ JSON เป็น Python Dictionary
                result = json.loads(json_response_text)
                
                analysis_summary = result.get('analysis_summary', {})
                keyword_frequency = result.get('keyword_frequency', [])
                
                # ตรวจสอบความสมบูรณ์ของคีย์หลัก
                if not analysis_summary or not keyword_frequency:
                    st.warning("⚠️ ผลลัพธ์ JSON ไม่สมบูรณ์: อาจขาด 'analysis_summary' หรือ 'keyword_frequency'")
                    st.code(json_response_text)
                    raise ValueError("JSON Incomplete") 

                st.success("✅ การวิเคราะห์เสร็จสมบูรณ์")
                
                # --- 3.5 การแสดงผลลัพธ์ ---
                
                ## 1. ตารางสรุปและวิเคราะห์โทน (Summary & Tone)
                st.header("1. สรุปและวิเคราะห์ภาพรวม")
                st.markdown("สามารถคลิกช่องในตารางเพื่อขยายดูเนื้อหาทั้งหมด")
                
                summary_df = pd.DataFrame({
                    "Summary": [analysis_summary.get('summary_text', 'N/A')],
                    "Tone Analysis": [analysis_summary.get('tone_analysis', 'N/A')],
                    "Readability Level": [analysis_summary.get('readability_level', 'N/A')]
                })
                
                # สร้าง DataFrame ที่ Transpose สำหรับแสดงผลและดาวน์โหลด
                summary_df_transposed = summary_df.T.rename(columns={0: "ผลการวิเคราะห์"})
                
                st.dataframe(
                    summary_df_transposed, 
                    use_container_width=True, 
                    height=200 # ใช้ Transpose เพื่อให้ดูง่ายขึ้น
                )
                
                st.markdown("---")
                
                ## 2. ตารางความถี่คำศัพท์สำคัญ (Keyword Frequency)
                st.header(f"2. การวิเคราะห์ความถี่คำศัพท์สำคัญ (Top {top_n_keywords} คำ)")
                
                frequency_df = pd.DataFrame(keyword_frequency)
                frequency_df.columns = ['คำศัพท์', 'ความถี่ที่ปรากฏ', 'ส่วนของคำพูด (POS)']
                frequency_df.insert(0, 'ลำดับ', range(1, 1 + len(frequency_df)))

                st.dataframe(
                    frequency_df, 
                    hide_index=True, 
                    use_container_width=True
                ) 

                st.header("3. แผนภูมิความถี่คำศัพท์")

                # ใช้ st.bar_chart เพื่อแสดงผลลัพธ์ความถี่คำศัพท์
                # โดยกำหนดให้แกน X เป็น 'คำศัพท์' และแกน Y เป็น 'ความถี่ที่ปรากฏ'
                st.bar_chart(
                    frequency_df.set_index('คำศัพท์')[['ความถี่ที่ปรากฏ']], 
                    use_container_width=True
                )
                
                # --- 3.6 การดาวน์โหลดผลลัพธ์  ---
                st.markdown("---")
                st.header("3. ดาวน์โหลดผลลัพธ์")
                
                # 1. Download Summary CSV (ใช้ summary_df_transposed ที่ถูกต้องแล้ว)
                summary_csv = summary_df_transposed.to_csv(index=True, encoding='utf-8')
                st.download_button(
                    label="⬇️ ดาวน์โหลดสรุปผลการวิเคราะห์ (CSV)",
                    data=summary_csv,
                    file_name='summary_analysis.csv',
                    mime='text/csv',
                    key='download_summary_csv'
                )

                # 2. Download Frequency CSV 
                frequency_csv = frequency_df.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="⬇️ ดาวน์โหลดตารางความถี่คำศัพท์ (CSV)",
                    data=frequency_csv,
                    file_name='keyword_frequency.csv',
                    mime='text/csv',
                    key='download_frequency_csv'
                )
                
            except (json.JSONDecodeError, ValueError) as e:
                # Catch JSON Decode Error และ Value Error
                st.error(f"❌ Error: AI ไม่ได้ส่งผลลัพธ์ในรูปแบบ JSON ที่ถูกต้อง หรือ JSON ไม่สมบูรณ์: {e}")
                st.markdown("**ผลลัพธ์ดิบที่ได้รับ:**")
                st.code(json_response_text)
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")






