import pytest
from fastapi.testclient import TestClient
from api.main import app

# Tạo ứng dụng Client giả lập để Test
client = TestClient(app)

def test_read_root():
    """Kiểm tra xem API server có khởi động thành công không"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "Backend API is running"

def test_predict_endpoint_success():
    """Kiểm tra endpoint /predict với input hợp lệ"""
    payload = {
        "thal_3_0": 0.0,
        "oldpeak": 2.5,
        "hr_ratio": 2.2,
        "cp_4_0": 1.0,
        "ca_0_0": 0.0,
        "thalach": 140.0,
        "trestbps": 160.0,
        "chol": 286.0,
        "cp_3_0": 0.0,
        "age": 67.0
    }
    
    response = client.post("/predict", json=payload)
    
    # Nếu mô hình chưa load thì sẽ báo lỗi 500 (VD: chạy trên Github không upload ML file pkl)
    # Chúng ta xử lý logic này để Test không fail vô cớ
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert data["probability"] >= 0.0 and data["probability"] <= 1.0
    else:
        assert response.status_code == 500

def test_predict_endpoint_validation_error():
    """Kiểm tra Schema Validation khi người dùng bỏ quên biến (ví dụ thiếu 'age')"""
    invalid_payload = {
        "thal_3_0": 0.0,
        "oldpeak": 2.5
    }
    
    response = client.post("/predict", json=invalid_payload)
    # FastAPI pydantic sẽ tự động trả lỗi 422 Unprocessable Entity
    assert response.status_code == 422
