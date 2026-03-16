import os
import streamlit as st
from google import genai

global_client = None

def configure_llm(api_key):
    global global_client
    if api_key:
        try:
            global_client = genai.Client(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Lỗi khởi tạo API Client: {e}")
            return False
    return False

def get_medical_advice(raw_input_data, risk_prob):
    if not global_client:
        return "Lỗi: Chưa có API Key hợp lệ được cấu hình. Vui lòng kiểm tra file .env hoặc cấu hình hệ thống."
        
    try:
        prompt = f"""
        Đóng vai một bác sĩ tim mạch chuyên khoa. Hệ thống AI cốt lõi (XGBoost) đã đánh giá bệnh nhân này có tỉ lệ rủi ro mắc bệnh tim mạch là {risk_prob:.2f}%.
        Đây là các thông số lâm sàng của bệnh nhân:
        {raw_input_data}
        
        Nhiệm vụ:
        1. Đưa ra một phân tích ngắn gọn (khoảng 3-4 câu) giải thích với bác sĩ đồng nghiệp vì sao ca này rủi ro cao/thấp.
        2. Gợi ý 2 bước tiếp theo y khoa (ví dụ xét nghiệm thêm gì).
        3. Viết bằng tiếng Việt, giọng điệu chuyên môn, khách quan.
        
        Không disclaimer dài dòng như 'tôi là AI', đi thẳng vào vấn đề.
        """
        
        response = global_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Lỗi trong quá trình chạy Model từ Google GenAI API: {e}. Vui lòng thử lại sau hoặc kiểm tra kết nối."

def get_patient_care_plan(raw_input_data, risk_prob):
    if not global_client:
        return "Lỗi: Chưa có API Key hợp lệ được cấu hình."
        
    try:
        prompt = f"""
        Đóng vai một bác sĩ dinh dưỡng và điều trị. Hệ thống đánh giá một bệnh nhân (rủi ro tim mạch {risk_prob:.2f}%) với số đo:
        {raw_input_data}
        
        Hãy viết một "Phác đồ lối sống ngắn gọn" cho chính bệnh nhân này đọc.
        - Giọng điệu: Đồng cảm, khuyên răn, dễ hiểu.
        - Gồm 3 gạch đầu dòng: (1) Chế độ ăn uống phù hợp với Cholesterol/Đường huyết của họ. (2) Vận động luyện tập. (3) Lưu ý rủi ro.
        """
        
        response = global_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Lỗi không thể tạo Care Plan từ GenAI: {e}."
