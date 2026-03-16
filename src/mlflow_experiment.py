"""
MLflow Experiment Tracking cho dự án Heart Disease Diagnosis.

Script này chạy huấn luyện TẤT CẢ 4 model (XGBoost, AdaBoost, Gradient Boosting, Random Forest),
ghi lại toàn bộ Hyperparameters + Metrics + Model Artifacts vào MLflow Tracking Server.

Cách chạy:
  1. Khởi động MLflow UI:    mlflow ui --port 5000
  2. Chạy thí nghiệm:        python src/mlflow_experiment.py
  3. Mở trình duyệt:         http://localhost:5000
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn

# ==========================================
# CẤU HÌNH
# ==========================================
SPLITS_DIR = Path('../splits')
MODELS_DIR = Path('../models')
TARGET = 'target'
DATASET_PREFIX = 'fe_dt'
EXPERIMENT_NAME = "Heart_Disease_Model_Comparison"

# ==========================================
# HÀM TIỆN ÍCH
# ==========================================
def load_data(prefix, split_type):
    file_path = SPLITS_DIR / f"{prefix}_{split_type}.csv"
    df = pd.read_csv(file_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def compute_metrics(model, X, y):
    """Tính toàn bộ metrics chuẩn ML"""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_proba)
    return metrics

# ==========================================
# ĐỊNH NGHĨA CÁC THÍ NGHIỆM
# ==========================================
experiments = [
    {
        "name": "XGBoost",
        "model": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=42, eval_metric='logloss', n_jobs=-1
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "algorithm": "XGBoost"},
        "pkl_name": "xgboost_model.pkl"
    },
    {
        "name": "AdaBoost",
        "model": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "algorithm": "AdaBoost"},
        "pkl_name": "adaboost_model.pkl"
    },
    {
        "name": "GradientBoosting",
        "model": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "algorithm": "GradientBoosting"},
        "pkl_name": "gradient_boosting_model.pkl"
    },
    {
        "name": "RandomForest",
        "model": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        "params": {"n_estimators": 100, "max_depth": 5, "algorithm": "RandomForest"},
        "pkl_name": "random_forest_model.pkl"
    },
]

# ==========================================
# MAIN: CHẠY THÍ NGHIỆM VỚI MLFLOW
# ==========================================
def main():
    # Thiết lập MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 60)
    print("MLFLOW EXPERIMENT TRACKING - HEART DISEASE MODEL COMPARISON")
    print("=" * 60)

    # Load dữ liệu
    X_train, y_train = load_data(DATASET_PREFIX, 'train')
    X_val, y_val = load_data(DATASET_PREFIX, 'val')
    X_test, y_test = load_data(DATASET_PREFIX, 'test')

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}\n")

    best_model_name = None
    best_f1 = 0.0

    for exp in experiments:
        print(f"\n--- Đang chạy thí nghiệm: {exp['name']} ---")

        # Mỗi lần train 1 model = 1 "Run" trong MLflow
        with mlflow.start_run(run_name=exp["name"]):

            # 1. Log Hyperparameters (Siêu tham số)
            mlflow.log_params(exp["params"])

            # 2. Huấn luyện Model
            model = exp["model"]
            model.fit(X_train, y_train)

            # 3. Tính Metrics trên từng tập
            train_metrics = compute_metrics(model, X_train, y_train)
            val_metrics = compute_metrics(model, X_val, y_val)
            test_metrics = compute_metrics(model, X_test, y_test)

            # 4. Log Metrics vào MLflow
            for k, v in train_metrics.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            # 5. Log Model Artifact (lưu bản sao model vào MLflow)
            mlflow.sklearn.log_model(model, artifact_path="model")

            # 6. Lưu file .pkl ra thư mục models/ (để FastAPI dùng)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, MODELS_DIR / exp["pkl_name"])

            # In kết quả
            print(f"  Train Accuracy: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1_score']:.4f}")
            print(f"  Val   Accuracy: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1_score']:.4f}")
            print(f"  Test  Accuracy: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_score']:.4f}")

            # Theo dõi model tốt nhất
            if test_metrics["f1_score"] > best_f1:
                best_f1 = test_metrics["f1_score"]
                best_model_name = exp["name"]

    print("\n" + "=" * 60)
    print(f"MODEL TỐT NHẤT: {best_model_name} (Test F1-Score: {best_f1:.4f})")
    print("=" * 60)
    print("\nMở trình duyệt http://localhost:5000 để xem Dashboard MLflow!")

if __name__ == "__main__":
    main()
