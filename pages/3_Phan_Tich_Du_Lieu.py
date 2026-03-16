import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Phân Tích Dữ Liệu", layout="wide")

st.title("📈 BẢNG ĐIỀU KHIỂN DỮ LIỆU EDA (Exploratory Data Analysis)")
st.markdown("Số liệu Cleveland Data Gốc được dùng trong phân tích:")

DATA_PATH = "data/cleveland.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        # Tránh lỗi parse do file cleveland
        try:
            return pd.read_csv(DATA_PATH, header=None) # Thông thường Cleveland k có header
        except:
            pass
    return pd.DataFrame()

df = load_data()

if not df.empty:
    with st.expander("👁️ Xem bảng dữ liệu thô"):
        st.dataframe(df.head(50))
        
    st.info(f"Kích thước tập dữ liệu huấn luyện: {df.shape[0]} hàng và {df.shape[1]} cột.")
    
    st.markdown("---")
    st.subheader("1. Phân bố Tuổi")
    if 0 in df.columns:
        fig_age = px.histogram(df, x=0, nbins=20, title="Biểu đồ Mật độ theo Tuổi bệnh nhân gốc",
                               labels={'0':'Tuổi', 'count': 'Số người'})
        st.plotly_chart(fig_age)
    
    st.subheader("2. Tương quan (Minh họa Scatter Plot)")
    if 0 in df.columns and 3 in df.columns:
        fig_scatter = px.scatter(df, x=0, y=3, color=13 if 13 in df.columns else None,
                                 title="Tuổi vs. Huyết Áp",
                                 labels={'0':'Tuổi', '3': 'Huyết áp (trestbps)'})
        st.plotly_chart(fig_scatter)
else:
    st.warning("Không tìm thấy file cleveland.csv để hiển thị EDA.")
