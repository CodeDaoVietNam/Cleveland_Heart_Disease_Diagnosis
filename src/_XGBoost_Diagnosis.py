import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)
import joblib

# ==========================================
# CẤU HÌNH & HẰNG SỐ
# ==========================================
# Cập nhật đường dẫn thư mục chứa dữ liệu
SPLITS_DIR = Path('../splits')
TARGET = 'target'

# Tùy chọn prefix dataset: 'raw', 'dt', 'fe', hoặc 'fe_dt'
DATASET_PREFIX = 'fe_dt'


# ==========================================
# CÁC HÀM TIỆN ÍCH
# ==========================================
def load_data(prefix, split_type):
    """
    Đọc dữ liệu phân chia (train/val/test) từ thư mục splits.
    """
    file_path = SPLITS_DIR / f"{prefix}_{split_type}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}. Vui lòng kiểm tra lại đường dẫn.")

    df = pd.read_csv(file_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate_model(model, X, y, split_name):
    """
    Đánh giá mô hình, in ra các chỉ số và trả về độ chính xác.
    """
    y_pred = model.predict(X)

    # Tính xác suất cho ROC AUC (nếu model hỗ trợ predict_proba)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_proba)
    else:
        roc_auc = None

    acc = accuracy_score(y, y_pred)

    print(f"\n{'=' * 15} ĐÁNH GIÁ TRÊN TẬP {split_name.upper()} {'=' * 15}")
    print(f"Accuracy : {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print("=" * 50)

    return acc


def plot_feature_importance(model, feature_names, top_n=10):
    """
    Trực quan hóa độ quan trọng của các đặc trưng (Feature Importances).
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances_sorted = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))

    # Sửa cảnh báo Seaborn: thêm hue=importances_sorted.index và legend=False
    sns.barplot(
        x=importances_sorted.values,
        y=importances_sorted.index,
        hue=importances_sorted.index,
        palette='viridis',
        legend=False
    )

    plt.title(f'Top {top_n} Feature Importances - XGBoost')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()

    # Khởi tạo thư mục figures nếu chưa có
    fig_dir = Path('../figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Lưu file trước khi show để tránh bị hình trắng
    file_path = fig_dir / 'xgboost_feature_importances.pdf'
    plt.savefig(file_path)
    print(f"\nĐã lưu biểu đồ Feature Importances thành '{file_path}'")

    # Hiển thị biểu đồ
    plt.show()


# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==========================================
def main():
    print(f"Đang tải dữ liệu '{DATASET_PREFIX}' từ {SPLITS_DIR}...")

    # 1. Tải dữ liệu
    X_train, y_train = load_data(DATASET_PREFIX, 'train')
    X_val, y_val = load_data(DATASET_PREFIX, 'val')
    X_test, y_test = load_data(DATASET_PREFIX, 'test')

    print(f"Kích thước tập Train: {X_train.shape}")
    print(f"Kích thước tập Val  : {X_val.shape}")
    print(f"Kích thước tập Test : {X_test.shape}")

    # 2. Khởi tạo và Huấn luyện mô hình XGBoost
    print("\nĐang huấn luyện mô hình XGBoost...")

    # Khởi tạo XGBClassifier (bạn có thể tinh chỉnh các tham số này)
    xgb_model = XGBClassifier(
        n_estimators=100,  # Số lượng cây
        learning_rate=0.1,  # Tốc độ học
        max_depth=3,  # Độ sâu tối đa của cây
        random_state=42,
        eval_metric='logloss',  # Metric dùng để đánh giá trong quá trình train (tránh warning)
        n_jobs=-1  # Dùng tất cả core CPU
    )

    xgb_model.fit(X_train, y_train)

    # 3. Đánh giá mô hình
    evaluate_model(xgb_model, X_train, y_train, 'Train')
    evaluate_model(xgb_model, X_val, y_val, 'Validation')
    evaluate_model(xgb_model, X_test, y_test, 'Test')

    # 4. Trực quan hóa Feature Importance
    plot_feature_importance(xgb_model, X_train.columns)

    # 5. Lưu mô hình
    model_dir = Path('../models')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_filename = model_dir / 'xgboost_model.pkl'
    joblib.dump(xgb_model, model_filename)
    print(f"\nĐã lưu mô hình đã huấn luyện vào file '{model_filename}'")


if __name__ == "__main__":
    main()