import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Nhập Liệu Hàng Loạt", layout="wide")

st.title("📂 CHUẨN ĐOÁN HÀNG LOẠT (BATCH PREDICTION)")
st.markdown("Tiết kiệm thời gian bằng việc tải lên danh sách bệnh nhân từ `CSV`.")

# Form tải lên file
uploaded_file = st.file_uploader("📂 Xin mời Upload file dữ liệu nhân viên khám (.CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load CSV
        df_input = pd.read_csv(uploaded_file)
        st.write("Dữ liệu thô vừa tải lên:")
        st.dataframe(df_input.head(5))

        st.info("Hệ thống tự động tiền xử lý (Auto-Feature Engineering)...")
        
        batch_payload = []
        for i, row in df_input.iterrows():
            age = row.get('age', 50)
            thalach = row.get('thalach', 150)
            patient_data = {
                'thal_3_0': 1.0 if row.get('thal') == 3.0 else 0.0,
                'oldpeak': float(row.get('oldpeak', 1.0)),
                'hr_ratio': float(thalach) / float(age) if age > 0 else 0.0,
                'cp_4_0': 1.0 if row.get('cp') == 4.0 else 0.0,
                'ca_0_0': 1.0 if row.get('ca') == 0.0 else 0.0,
                'thalach': float(thalach),
                'trestbps': float(row.get('trestbps', 120)),
                'chol': float(row.get('chol', 200)),
                'cp_3_0': 1.0 if row.get('cp') == 3.0 else 0.0,
                'age': float(age)
            }
            batch_payload.append(patient_data)
        
        # Dự đoán
        if st.button("🚀 BẮT ĐẦU CHẠY AI CẢ LÔ (RUN BATCH)", type="primary"):
            with st.spinner("Hết nối máy chủ API chạy chẩn đoán lớn..."):
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                
                res = requests.post(f"{backend_url}/batch_predict", json=batch_payload)
                if res.status_code == 200:
                    results = res.json()
                    
                    preds = [r['prediction'] for r in results]
                    probas = [r['probability'] for r in results]
                    
                    df_input['Dự Đoán Y Tế AI'] = ["Bị Bệnh (Risk=1)" if p == 1 else "Bình Thường (Risk=0)" for p in preds]
                    df_input['Xác Suất Bệnh (AI%)'] = [round(float(p)*100, 2) for p in probas]
                    
                    st.success("Hoàn thành! Kết quả chẩn đoán đã ghép vào bảng dữ liệu.")
                    
                    # Tô màu hiển thị đẹp
                    def highlight_risk(val):
                        color = '#ffccd5' if 'Bị Bệnh' in str(val) else '#e0fbfc'
                        return f'background-color: {color}'
                    
                    st.dataframe(df_input.style.applymap(highlight_risk, subset=['Dự Đoán Y Tế AI']))
                    
                    # Xuất ra CSV Report
                    csv = df_input.to_csv(index=False).encode('utf-8')
                    
                    st.download_button(
                        label="📥 Tải xuống danh sách Đã Chẩn Đoán Cả Lô (.CSV)",
                        data=csv,
                        file_name='Batch_Predict_Result_Heart_Disease.csv',
                        mime='text/csv',
                        type="primary"
                    )
                else:
                    st.error(f"Máy chủ API báo lỗi: {res.text}")

    except Exception as e:
        st.error(f"Lỗi đọc file hoặc tương thích dữ liệu: {e}")
        st.warning("Vui lòng tải lên file chứa cột nguyên bản: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.")
