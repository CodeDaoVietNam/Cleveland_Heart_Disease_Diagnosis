import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,roc_auc_score)
import joblib

# ==========================================
# CẤU HÌNH & HẰNG SỐ
# ==========================================
# Cập nhật đường dẫn như bạn yêu cầu
SPLITS_DIR =  Path('../splits')
TARGET = 'target'

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

    # 1. Sửa cảnh báo Seaborn: thêm hue=importances_sorted.index và legend=False
    sns.barplot(
        x=importances_sorted.values,
        y=importances_sorted.index,
        hue=importances_sorted.index,
        palette='viridis',
        legend=False
    )

    plt.title(f'Top {top_n} Feature Importances - Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    fig_dir = Path('../figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Lưu file
    file_path = fig_dir / 'rf_feature_importances.pdf'
    plt.savefig(file_path)
    print(f"\nĐã lưu biểu đồ Feature Importances thành '{file_path}'")

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

    # 2. Khởi tạo và Huấn luyện mô hình
    print("\nĐang huấn luyện mô hình Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Số lượng cây quyết định
        max_depth=7,  # Độ sâu tối đa của cây (chống overfitting)
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',  # Xử lý mất cân bằng dữ liệu nếu có
        n_jobs=-1  # Tận dụng tất cả luồng CPU để train nhanh hơn
    )

    rf_model.fit(X_train, y_train)

    # 3. Đánh giá mô hình
    evaluate_model(rf_model, X_train, y_train, 'Train')
    evaluate_model(rf_model, X_val, y_val, 'Validation')
    evaluate_model(rf_model, X_test, y_test, 'Test')

    # 4. Trực quan hóa Feature Importance
    plot_feature_importance(rf_model, X_train.columns)

    # 5. Lưu mô hình (Tùy chọn)
    model_dir = Path('../models')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_filename = model_dir/'random_forest_model.pkl'
    joblib.dump(rf_model, model_filename)
    print(f"\nĐã lưu mô hình đã huấn luyện vào file '{model_filename}'")


if __name__ == "__main__":
    main()
