import streamlit as st
import requests
import os

st.set_page_config(page_title="Tổng Quan Mô Hình", layout="wide")

st.title("📊 BIỂU ĐỒ BÁO CÁO KỸ THUẬT MÔ HÌNH")
st.markdown("Cái nhìn vào 'hộp đen' của hệ thống AI.")

st.subheader("Trọng số ảnh hưởng (Feature Importance)")
st.info("Biểu đồ thể hiện tính trạng nào trong y tế là nguyên nhân dẫn đến phán đoán bệnh cao nhất.")

with st.spinner("Đang truy xuất Feature Importance Image từ máy chủ Microservice Backend..."):
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    try:
        res = requests.get(f"{backend_url}/importance")
        if res.status_code == 200:
            st.image(res.content, use_container_width=True)
        else:
            st.error(f"Máy chủ API báo lỗi: {res.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi không thể kết nối Máy chủ API Backend: {e}")

st.markdown("---")
st.subheader("Bảng Chỉ số (Metrics) Báo cáo Huấn luyện (Minh họa)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", "88.2%", "+2.3% so với bản cũ")
col2.metric("Precision", "87.5%", "")
col3.metric("Recall (Sensitivity)", "91.1%", "Ưu tiên tránh sót bệnh")
col4.metric("F1-Score", "89.2%", "")

st.markdown("Hệ thống đã trải qua **Cross Validation (K-Fold = 5)** và tối ưu hóa siêu tham số bằng Bayesian Optimization.")
