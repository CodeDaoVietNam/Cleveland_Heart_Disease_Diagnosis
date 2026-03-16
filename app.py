import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import requests
import datetime
import os
import json
from dotenv import load_dotenv

# Import modules custom
from utils.db import init_db, save_patient_record
from utils.pdf_gen import generate_pdf_report
from utils.llm_assistant import configure_llm, get_medical_advice, get_patient_care_plan
from utils.ocr_helper import extract_medical_data_from_image
# Load .env
load_dotenv()
configure_llm(os.getenv("GEMINI_API_KEY"))

# Initialize Database
init_db()

# ==========================================
# CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Hệ Thống Chẩn Đoán Cấp Cao",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 40px;
        color: #E63946;
        text-align: center;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-text {
        font-size: 16px;
        color: #457B9D;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }
    .section-title {
        font-size: 18px;
        color: #1D3557;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #A8DADC;
    }
    .result-success {
        color: #2A9D8F;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background-color: #E9F5F3;
        border-radius: 12px;
        border: 1px solid #2A9D8F;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .result-danger {
        color: #E63946;
        font-size: 24px;
        font-weight: 800;
        text-align: center;
        padding: 20px;
        background-color: #FCEAEA;
        border-radius: 12px;
        border: 1px solid #E63946;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background-color: #E63946;
        color: white;
        height: 60px;
        font-size: 20px;
        font-weight: 800;
        border-radius: 12px;
        margin-top: 15px;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #D62828;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# HÀM LOAD DỮ LIỆU TỪ DB (không load mô hình nữa)
# ==========================================

def generate_patient_id():
    return f"PT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

def on_submit_calculation():
    # Only generate a new ID explicitly when the user clicks the calculate button
    st.session_state.patient_id = generate_patient_id()

# ==========================================
# GIAO DIỆN CHÍNH (TRANG 1)
# ==========================================
def main():
    st.markdown('<div class="main-header">❤️ CỔNG CHẨN ĐOÁN Y TẾ LÂM SÀNG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">AI Hỗ Trợ Đánh Giá Nguy Cơ Tim Mạch Chuyên Sâu (Enterprise Edition)</div>', unsafe_allow_html=True)

    # Sinh Patient ID ảo cho ca khám
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = generate_patient_id()

    st.info(f"🩺 ID Ca khám hiện tại: **{st.session_state.patient_id}**")

    # Nâng Cấp 3: UPLOAD BÁO CÁO XÉT NGHIỆM ĐỂ AUTO-FILL (Computer Vision)
    st.markdown('<div class="section-title">🖼️ Trí tuệ Thị giác (Auto-Fill Mẫu Xét Nghiệm)</div>', unsafe_allow_html=True)
    
    ocr_chol = 200.0
    ocr_trestbps = 120.0
    ocr_fbs_index = 1 # Mặc định là Bình Thường (Index 1 của List keys)
    
    uploaded_image = st.file_uploader("Upload ảnh Báo cáo / Phiếu xét nghiệm để tự động điền vào Form", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        with st.spinner("🔍 Hệ thống Vision EasyOCR đang quét phiếu y khoa..."):
            extracted_data, full_text = extract_medical_data_from_image(uploaded_image.read())
            
            if extracted_data.get('chol'):
                ocr_chol = float(extracted_data['chol'])
                st.success(f"✔️ Phát hiện Cholesterol: {ocr_chol}")
            if extracted_data.get('trestbps'):
                ocr_trestbps = float(extracted_data['trestbps'])
                st.success(f"✔️ Phát hiện Huyết áp (trestbps): {ocr_trestbps}")
            if 'fbs' in extracted_data:
                ocr_fbs_index = 0 if extracted_data['fbs'] == 1.0 else 1
                fbs_status = "Cao (> 120)" if extracted_data['fbs'] == 1.0 else "Bình Thường"
                st.success(f"✔️ Phát hiện Đường huyết lúc đói: {fbs_status}")
            
            if not extracted_data:
                st.warning("⚠️ Không tìm thấy số liệu (Cholesterol, Huyết áp, Đường huyết) trong ảnh. Vui lòng nhập tay.")
    st.markdown("---")

    # Bắt đầu form nhập liệu
    with st.form(key='patient_form'):
        st.markdown('<div class="section-title">🏥 NHẬP THÔNG TIN BỆNH NHÂN (LÂM SÀNG)</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1. THÔNG TIN CƠ BẢN**")
            age = st.number_input("Tuổi", min_value=1, max_value=120, value=50, step=1)
            
            sex_map = {"Nam": 1.0, "Nữ": 0.0}
            sex_input = st.selectbox("Giới tính", options=list(sex_map.keys()))
            sex = sex_map[sex_input]
            
            trestbps = st.number_input("Huyết áp lúc nghỉ (mmHg)", value=ocr_trestbps, step=1.0)
            # Inline Warning (Inline Guide nấc 1)
            if trestbps > 140:
                st.warning("⚠️ Chỉ số Huyết áp đang ở mức Cao (Cần lưu ý).")

            chol = st.number_input("Cholesterol huyết thanh (mg/dl)", value=ocr_chol, step=1.0)
            if chol > 240:
                st.warning("⚠️ Mức Cholesterol lọt ngưỡng rủi ro tắc nghẽn.")

        with col2:
            st.markdown("**2. CHỈ SỐ KHÁM LÂM SÀNG**")
            cp = st.selectbox("Loại đau ngực (CP)", options=[1.0, 2.0, 3.0, 4.0],
                              help="1: Điển hình, 2: Không điển hình, 3: Không do thắt ngực, 4: Không triệu chứng")
            if cp == 4.0:
                st.error("⚠️ Đau ngực Cấp độ 4: Có tính chất bất thường cực cao!")
            
            fbs_map = {"Cao (> 120 mg/dl)": 1.0, "Bình thường (≤ 120 mg/dl)": 0.0}
            fbs_input = st.selectbox("Đường huyết lúc đói (FBS)", options=list(fbs_map.keys()), index=ocr_fbs_index)
            fbs = fbs_map[fbs_input]
            
            restecg = st.selectbox("Điện tâm đồ lúc nghỉ (RestECG)", options=[0.0, 1.0, 2.0],
                                   help="0: Bình thường, 1: Bất thường sóng ST-T, 2: Phì đại thất trái có khả năng")

            exang_map = {"Có": 1.0, "Không": 0.0}
            exang_input = st.selectbox("Đau ngực sau gắng sức (Exang)", options=list(exang_map.keys()))
            exang = exang_map[exang_input]

        with col3:
            st.markdown("**3. KIỂM TRA GẮNG SỨC & KHÁC**")
            thalach = st.number_input("Nhịp tim tối đa (Thalach)", value=150.0, step=1.0)
            
            oldpeak = st.number_input("Độ giảm ST do gắng sức (Oldpeak)", value=1.0, step=0.1)
            slope = st.selectbox("Độ dốc đoạn ST (Slope)", options=[1.0, 2.0, 3.0],
                                 help="1: Đi lên, 2: Bằng phẳng, 3: Đi xuống")
            ca = st.selectbox("Số mạch máu nhuộm màu (CA)", options=[0.0, 1.0, 2.0, 3.0])
            thal = st.selectbox("Thalassemia (Thal)", options=[3.0, 6.0, 7.0],
                                help="3: Bình thường, 6: Khiếm khuyết cố định, 7: Khiếm khuyết có thể phục hồi")

        submit_button = st.form_submit_button(label='🔍 TINH TOÁN MỨC ĐỘ RỦI RO', type='primary', on_click=on_submit_calculation)

    if submit_button:
        raw_input_data = {
            "Tuổi": age, "Giới tính": "Nam" if sex == 1.0 else "Nữ", "Loại đau ngực": cp, 
            "Huyết áp": trestbps, "Cholesterol": chol, "Đường huyết cao": "Có" if fbs == 1.0 else "Không", 
            "Điện tâm đồ": restecg, "Nhịp tim max": thalach, "Đau ngực gắng sức": "Có" if exang == 1.0 else "Không", 
            "Suy giảm ST": oldpeak, "Độ dốc ST": slope, "Mạch máu": ca, "Thalassemia": thal
        }
        
        transformed_data = {
            'thal_3_0': 1.0 if thal == 3.0 else 0.0,
            'oldpeak': float(oldpeak),
            'hr_ratio': float(thalach) / float(age) if age > 0 else 0.0,
            'cp_4_0': 1.0 if cp == 4.0 else 0.0,
            'ca_0_0': 1.0 if ca == 0.0 else 0.0,
            'thalach': float(thalach),
            'trestbps': float(trestbps),
            'chol': float(chol),
            'cp_3_0': 1.0 if cp == 3.0 else 0.0,
            'age': float(age)
        }
        
        try:
            with st.spinner("⏳ Đang gửi dữ liệu lên Backend API..."):
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                
                try:
                    response = requests.post(f"{backend_url}/predict", json=transformed_data)
                    response.raise_for_status()
                except requests.exceptions.RequestException as req_err:
                    st.error(f"❌ Không thể kết nối tới Backend API. Vui lòng kiểm tra server. Chi tiết lỗi: {req_err}")
                    return
                
                res_json = response.json()
                prediction = res_json.get('prediction', 0)
                probability = res_json.get('probability', 0.0)

                # Lưu vào DB (Nấc 3 - Item 6)
                save_patient_record(st.session_state.patient_id, raw_input_data, prediction, probability)

            st.markdown("---")
            st.markdown('<div class="section-title">📋 HỒ SƠ Y TẾ TRẢ VỀ (MEDICAL ASSESSMENT)</div>', unsafe_allow_html=True)
            
            col_res1, col_res2 = st.columns([1.2, 1])

            with col_res1:
                if prediction == 1:
                    st.markdown('<div class="result-danger">⚠️ PHÁT HIỆN RỦI RO LỚN TỪ HỆ THỐNG KIỂM TRA</div>', unsafe_allow_html=True)
                    st.error("🩺 **KHUYẾN CÁO Y TẾ:** Bệnh nhân thể hiện nhiều thông số tương đồng với bệnh lý tim mạch, cần sắp xếp khám và tầm soát chuyên sâu ngay.")
                else:
                    st.markdown('<div class="result-success">✅ KẾT QUẢ KIỂM TRA KHÔNG ĐÁNG CẢNH BÁO</div>', unsafe_allow_html=True)
                    st.success("🩺 **KHUYẾN CÁO Y TẾ:** Bệnh nhân hiện ở mức an toàn. Duy trì lối sống khỏe mạnh và có thể tái khám định kỳ 6 tháng một lần.")
                
                # Nâng Cấp 5: Không hiển thị Heart Animation nữa

            with col_res2:
                # Gauge Chart
                if probability < 0.4:
                    bar_color = "#2A9D8F"
                elif probability < 0.7:
                    bar_color = "#F4A261"
                else:
                    bar_color = "#E63946"
                    
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={'suffix': "%", 'font': {'size': 40, 'color': bar_color}},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "TỈ LỆ RỦI RO (HỆ THỐNG ƯỚC LƯỢNG)", 'font': {'size': 14, 'color': '#1D3557'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': bar_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(42, 157, 143, 0.2)"},
                            {'range': [40, 70], 'color': "rgba(244, 162, 97, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(230, 57, 70, 0.2)"}],
                        'threshold': {'line': {'color': "#1D3557", 'width': 3}, 'thickness': 0.75, 'value': probability * 100}
                    }
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
                st.plotly_chart(fig)

            # --- MẶT CẮT TRỌNG TÂM: SHAP / EXPLAINABLE AI --- (Nấc 2)
            st.markdown("---")
            st.markdown('<div class="section-title">🧠 TÍNH NĂNG XAI - GIẢI THÍCH CHI TIẾT VÌ SAO AI RA QUYẾT ĐỊNH NÀY?</div>', unsafe_allow_html=True)
            st.write("Biểu đồ Waterfall phía dưới bóc tách rõ từng yếu tố khiến rủi ro tăng / giảm như thế nào. Màu đỏ là nguyên nhân tăng rủi ro, Màu xanh lá làm giảm rủi ro.")
            
            with st.spinner("Đang render Waterfall Explainer bằng SHAP từ API Server..."):
                try:
                    shap_res = requests.post(f"{backend_url}/explain", json=transformed_data)
                    if shap_res.status_code == 200:
                        st.image(shap_res.content, use_container_width=True)
                    else:
                        st.warning("⚠️ Không thể lấy báo cáo XAI từ Backend ở lúc này.")
                except requests.exceptions.RequestException:
                    st.warning("⚠️ Lỗi mạng, không kết nối được tính năng XAI của hệ thống Backend.")

            # --- Nâng cấp 1 & 2: TRỢ LÝ Y KHOA LLM ---
            st.markdown("---")
            st.markdown('<div class="section-title">🤖 TRỢ LÝ BÁC SĨ ẢO (LLM GENERATIVE AI)</div>', unsafe_allow_html=True)
            
            # API Key Configuration Warning
            if not os.getenv("GEMINI_API_KEY"):
                st.warning("⚠️ Chưa cấu hình 'GEMINI_API_KEY' trong file .env. Bạn cần xin 1 API key từ Google AI Studio (Miễn phí) và ghi vào file .env ở thư mục gốc để Trợ lý AI có thể hoạt động.")
            else:
                with st.spinner("Đang kết nối siêu AI để biên soạn phác đồ tư vấn..."):
                    llm_advice = get_medical_advice(raw_input_data, probability * 100)
                    llm_care_plan = get_patient_care_plan(raw_input_data, probability * 100)
                
                tab_advice, tab_patient = st.tabs(["👨‍⚕️ Góc Bác Sĩ (Tư vấn Nội bộ)", "👤 Góc Bệnh Nhân (Phác đồ đi kèm)"])
                
                with tab_advice:
                    st.info(llm_advice)
                
                with tab_patient:
                    st.success(llm_care_plan)

            # --- Nấc 1, Item 2: REPORT BÀN GIAO CHUYÊN MÔN PDF ---
            st.markdown("---")
            st.markdown('<div class="section-title">📥 TRÍCH XUẤT HỒ SƠ Y TẾ PDF</div>', unsafe_allow_html=True)
            pdf_bytes = generate_pdf_report(st.session_state.patient_id, raw_input_data, prediction, probability)
            
            st.download_button(
                label="📁 KẾT XUẤT HỒ SƠ BỆNH ÁN (PDF format)",
                data=pdf_bytes,
                file_name=f"Hospital_Cardio_Report_{st.session_state.patient_id}.pdf",
                mime="application/pdf",
                type="primary"
            )

            # ID will be reset next time the user clicks "Tính toán mức độ rủi ro" due to the on_click callback.

        except Exception as e:
            st.error(f"❌ Đã xảy ra lỗi hệ thống trong khối xử lý lõi: {e}")

if __name__ == '__main__':
    main()