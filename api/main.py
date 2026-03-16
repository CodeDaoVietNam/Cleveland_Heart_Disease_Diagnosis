from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
from api.auth import (
    authenticate_user, create_access_token, get_current_user,
    Token, UserOut
)
import joblib
import pandas as pd
from pathlib import Path
import io
import shap
import matplotlib
from utils.logger import get_logger
matplotlib.use('Agg') # Necessary for drawing plots outside the main thread
import matplotlib.pyplot as plt

app = FastAPI(title="Heart Disease E2E API", version="1.0.0")
logger = get_logger("fastapi-backend")

# Load compiled ML model
model_path = Path('./models/xgboost_model.pkl')
model = None
if model_path.exists():
    model = joblib.load(model_path)
    logger.info("Đã tải thành công hệ thống Machine Learning (xgboost_model.pkl) vào bộ nhớ RAM.")
else:
    logger.error("NGHIÊM TRỌNG: Máy chủ không tìm thấy file xgboost_model.pkl tại thư mục models/ !")

# Pydantic schema for strict payload validation
class PatientInput(BaseModel):
    thal_3_0: float
    oldpeak: float
    hr_ratio: float
    cp_4_0: float
    ca_0_0: float
    thalach: float
    trestbps: float
    chol: float
    cp_3_0: float
    age: float

@app.get("/")
def read_root():
    return {"status": "Backend API is running", "model_loaded": model is not None}

# ==========================================
# AUTHENTICATION ENDPOINTS (PUBLIC)
# ==========================================
@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Đăng nhập: Gửi username + password, nhận về JWT Token.
    Dùng token này gắn vào Header để gọi các API bảo mật.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Sai tên đăng nhập hoặc mật khẩu.")
    access_token = create_access_token(data={"sub": user["username"]})
    logger.info(f"AUTH :: Người dùng '{user['username']}' đăng nhập thành công.")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserOut)
def read_users_me(current_user: dict = Depends(get_current_user)):
    """Xem thông tin tài khoản hiện tại (cần token)"""
    return UserOut(**current_user)

# ==========================================
# PROTECTED ENDPOINTS (CẦN TOKEN)
# ==========================================

@app.post("/predict")
def predict_heart_disease(patient: PatientInput, current_user: dict = Depends(get_current_user)):
    """
    Inference endpoint: Accepts JSON payload, returns prediction score
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Máy chủ chưa nạp được mô hình ML.")
    
    # Map fields dynamically for the XGBoost features array
    features = ['thal_3.0', 'oldpeak', 'hr_ratio', 'cp_4.0', 'ca_0.0', 'thalach', 'trestbps', 'chol', 'cp_3.0', 'age']
    data = [[
        patient.thal_3_0, patient.oldpeak, patient.hr_ratio, patient.cp_4_0, patient.ca_0_0,
        patient.thalach, patient.trestbps, patient.chol, patient.cp_3_0, patient.age
    ]]
    df = pd.DataFrame(data, columns=features)
    
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else 0.0
    
    logger.info(f"API /predict :: Khách hàng {patient.age} tuổi - KQ={prediction} - XS={probability:.2f}")
    
    return {
        "prediction": prediction, 
        "probability": probability,
        "message": "Risk found" if prediction == 1 else "Normal"
    }

@app.post("/explain")
def explain_heart_disease(patient: PatientInput, current_user: dict = Depends(get_current_user)):
    """
    Explainability endpoint: Returns a SHAP waterfall plot as an image
    """
    if model is None:
        logger.error("API /explain :: Lỗi không chạy được (mô hình None)")
        raise HTTPException(status_code=500, detail="Mô hình chưa được nạp.")
    
    logger.info("API /explain :: Hệ thống đang kết xuất hình ảnh phân tích SHAP...")
    features = ['thal_3.0', 'oldpeak', 'hr_ratio', 'cp_4.0', 'ca_0.0', 'thalach', 'trestbps', 'chol', 'cp_3.0', 'age']
    data = [[
        patient.thal_3_0, patient.oldpeak, patient.hr_ratio, patient.cp_4_0, patient.ca_0_0,
        patient.thalach, patient.trestbps, patient.chol, patient.cp_3_0, patient.age
    ]]
    df = pd.DataFrame(data, columns=features)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/batch_predict")
def batch_predict(patients: List[PatientInput], current_user: dict = Depends(get_current_user)):
    """
    Nhận 1 danh sách dữ liệu bệnh nhân và tính toán trả về hàng loạt
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Máy chủ chưa nạp được mô hình ML.")
    
    features = ['thal_3.0', 'oldpeak', 'hr_ratio', 'cp_4.0', 'ca_0.0', 'thalach', 'trestbps', 'chol', 'cp_3.0', 'age']
    data = [[
        p.thal_3_0, p.oldpeak, p.hr_ratio, p.cp_4_0, p.ca_0_0,
        p.thalach, p.trestbps, p.chol, p.cp_3_0, p.age
    ] for p in patients]
    df = pd.DataFrame(data, columns=features)
    
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1] if hasattr(model, "predict_proba") else [0.0]*len(preds)
    
    results = []
    for p, pr in zip(preds, probas):
        results.append({
            "prediction": int(p),
            "probability": float(pr)
        })
    return results

@app.get("/importance")
def get_feature_importance():
    """
    Trả về biểu đồ feature importance dưới dạng File Ảnh
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Mô hình chưa được nạp.")
    
    if hasattr(model, 'get_booster'):
        import xgboost as xgb
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(model.get_booster(), ax=ax, importance_type='weight', max_num_features=10)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return Response(content=buf.getvalue(), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Mô hình không hỗ trợ xuất biểu đồ importance plot.")
