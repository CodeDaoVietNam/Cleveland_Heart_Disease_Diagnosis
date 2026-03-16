import streamlit as st
import pandas as pd
from utils.db import get_all_records

st.set_page_config(page_title="Lịch Sử Bệnh Án", layout="wide")

st.title("🕒 HỒ SƠ LƯU TRỮ Y TẾ BỆNH NHÂN")
st.markdown("Xem lại nhật ký chẩn đoán qua hệ thống chuyên sâu.")

records_df = get_all_records()

if not records_df.empty:
    st.info(f"Hệ thống Database lưu trữ {len(records_df)} phiên làm việc.")
    
    # Định dạng cột cho đẹp, tính % xác suất
    display_df = records_df.copy()
    display_df['Xác suất Rủi ro'] = display_df['probability'].apply(lambda x: f"{x*100:.2f}%")
    display_df['Kết quả (AI)'] = display_df['prediction'].apply(lambda x: "⚠️ Nguy Cơ TIM" if x == 1 else "✅ Bình thường")
    
    # Đổi vị trí cột cho bác sĩ dễ xem
    cols_order = ['id', 'patient_id', 'diagnosis_date', 'Kết quả (AI)', 'Xác suất Rủi ro', 
                  'age', 'sex', 'trestbps', 'chol', 'cp', 'thalach']
    display_df = display_df[[c for c in cols_order if c in display_df.columns]]
    
    st.dataframe(display_df)
    
    st.markdown("---")
    st.subheader("🔍 Tìm Kiếm Bệnh Nhân")
    search_id = st.text_input("Nhập ID Patient (Ví dụ PT-2023...)")
    if search_id:
        result = display_df[display_df['patient_id'].str.contains(search_id, case=False, na=False)]
        if not result.empty:
            st.success(f"Tìm thấy {len(result)} hồ sơ khớp {search_id}")
            st.dataframe(result)
        else:
            st.warning("Không tìm thấy mã bệnh nhân này trong Lưu Trữ Đám Mây Mảnh!")
else:
    st.warning("Chưa có phiên làm việc (Khám Bệnh) nào được lưu lại trong Database.")
    st.markdown("Hãy sang trang **1. Chẩn Đoán Lâm Sàng** và nhấn tính điểm để bản rủi ro được lưu trữ vào CSDL.")
