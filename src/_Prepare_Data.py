import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler , OneHotEncoder, MinMaxScaler

SEED = 42
DATA_PATH = '../data/cleveland.csv'
COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
TARGET = 'target'
K_FEATURES = 10
OUT_DIR = Path('../splits')

NUMERIC_COLS = ['age','trestbps','chol','thalach','oldpeak']
CATEGORICAL_COLS = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

# ==========================================
# CÁC HÀM TIỆN ÍCH
# ==========================================
def set_seed(seed=SEED):
    """Thiết lập seed để đảm bảo kết quả có thể tái lập."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Seed set to: {seed}")

def load_and_clean_data(filepath):
    """Đọc và làm sạch dữ liệu cơ bản."""
    raw = pd.read_csv(filepath)
    raw.columns = COLUMNS

    cols_to_numeric = ['age','trestbps','chol','thalach','oldpeak','ca','thal']
    for  c in cols_to_numeric:
        raw[c] = pd.to_numeric(raw[c],errors = 'coerce')

    #Binarize target
    raw['target'] = (raw['target']>0).astype(int)
    print(f"Raw data shape: {raw.shape}")
    return raw

def split_data(df):
    """Chia dữ liệu thành Train, Validation và Test (60/20/20)."""
    raw_feature_cols = [c for c in df.columns if c != TARGET]
    X_all = df[raw_feature_cols]
    y_all = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_dataset(X_tr, y_tr, X_va,y_va,X_te,y_te,prefix):
    """Hàm tiện ích để lưu các bản split ra file CSV."""
    pd.concat([X_tr,y_tr.rename(TARGET)],axis = 1).to_csv(OUT_DIR / f'{prefix}_train.csv', index=False)
    pd.concat([X_va, y_va.rename(TARGET)], axis=1).to_csv(OUT_DIR / f'{prefix}_val.csv', index=False)
    pd.concat([X_te, y_te.rename(TARGET)], axis=1).to_csv(OUT_DIR / f'{prefix}_test.csv', index=False)
    print(f"Saved {prefix} splits.")

# ==========================================
# FEATURE ENGINEERING CLASSES & FUNCTIONS
# ==========================================
def add_new_features_func(df):
    df = df.copy()
    if {'chol', 'age'} <= set(df.columns):
        df['chol_per_age'] = df['chol'] / df['age']
    if {'trestbps', 'age'} <= set(df.columns):
        df['bps_per_age'] = df['trestbps'] / df['age']
    if {'thalach', 'age'} <= set(df.columns):
        df['hr_ratio'] = df['thalach'] / df['age']
    if 'age' in df.columns:
        df['age_bin'] = pd.cut(df['age'], bins=5, labels=False).astype('category')
    return df

class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        self.columns_ = X.columns
        self.new_features_ = []
        if {'chol', 'age'} <= set(X.columns):
            self.new_features_.append('chol_per_age')
        if {'trestbps', 'age'} <= set(X.columns):
            self.new_features_.append('bps_per_age')
        if {'thalach', 'age'} <= set(X.columns):
            self.new_features_.append('hr_ratio')
        if 'age' in X.columns:
            self.new_features_.append('age_bin')
        return self

    def transform(self, X):
        return add_new_features_func(X)

    def get_feature_names_out(self, input_features=None):
        return list(self.columns_) + self.new_features_


# ==========================================
# CHƯƠNG TRÌNH CHÍNH (MAIN)
# ==========================================
def main():
    # 1. Setup môi trường
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 2. Đọc và chia dữ liệu
    raw_df = load_and_clean_data(DATA_PATH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(raw_df)
    #3. Data post-processing (Raw Pipeline)
    cat_proc_raw = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', MinMaxScaler())
    ])
    num_proc_raw = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocess_raw = ColumnTransformer([
        ('num', num_proc_raw, NUMERIC_COLS),
        ('cat', cat_proc_raw, CATEGORICAL_COLS),
    ])

    raw_pipeline = Pipeline([('preprocess', preprocess_raw)])

    X_raw_train = raw_pipeline.fit_transform(X_train, y_train)
    X_raw_val = raw_pipeline.transform(X_val)
    X_raw_test = raw_pipeline.transform(X_test)

    # Trích xuất tên cột sau preprocess
    preprocessed_feature_names = []
    for name, transformer, columns in preprocess_raw.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            preprocessed_feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            preprocessed_feature_names.extend(columns)

    X_raw_train_df = pd.DataFrame(X_raw_train, columns=preprocessed_feature_names, index=X_train.index)
    X_raw_val_df = pd.DataFrame(X_raw_val, columns=preprocessed_feature_names, index=X_val.index)
    X_raw_test_df = pd.DataFrame(X_raw_test, columns=preprocessed_feature_names, index=X_test.index)

    save_dataset(X_raw_train_df, y_train, X_raw_val_df, y_val, X_raw_test_df, y_test, 'raw')

    # 4. Decision Tree Feature Selection (trên Raw Data)
    dt_selection_pipeline = Pipeline([
        ('preprocess', preprocess_raw),
        ('decision_tree', DecisionTreeClassifier(random_state=SEED))
    ])
    dt_selection_pipeline.fit(X_raw_train_df, y_train)

    fi_raw = pd.Series(
        dt_selection_pipeline.named_steps['decision_tree'].feature_importances_,
        index=preprocessed_feature_names
    ).sort_values(ascending=False)
    selected_raw_features = fi_raw.head(K_FEATURES).index.tolist()
    print(f"\nTop {K_FEATURES} selected RAW features by DT: {selected_raw_features}")

    X_dt_train = X_raw_train_df[selected_raw_features]
    X_dt_val = X_raw_val_df[selected_raw_features]
    X_dt_test = X_raw_test_df[selected_raw_features]
    save_dataset(X_dt_train, y_train, X_dt_val, y_val, X_dt_test, y_test, 'dt')
    # 5. Feature Engineering Pipeline
    gen_num = ['chol_per_age', 'bps_per_age', 'hr_ratio']
    gen_cat = ['age_bin']
    all_nums = NUMERIC_COLS + gen_num
    all_cats = CATEGORICAL_COLS + gen_cat

    num_proc_fe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler())
    ])
    cat_proc_fe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocess_fe = ColumnTransformer([
        ('num', num_proc_fe, all_nums),
        ('cat', cat_proc_fe, all_cats),
    ], verbose_feature_names_out=False).set_output(transform='pandas')

    fe_pipeline = Pipeline([
        ('add', AddNewFeaturesTransformer()),
        ('pre', preprocess_fe),
    ]).set_output(transform='pandas')

    Xt_tr = fe_pipeline.fit_transform(X_train, y_train)
    Xt_va = fe_pipeline.transform(X_val)
    Xt_te = fe_pipeline.transform(X_test)

    # Loại bỏ các cột có variance = 0 (chỉ có 1 giá trị duy nhất)
    nz_cols = Xt_tr.columns[Xt_tr.nunique(dropna=False) > 1]
    Xt_tr = Xt_tr[nz_cols]
    Xt_va = Xt_va[nz_cols]
    Xt_te = Xt_te[nz_cols]

    # 6. Mutual Information cho FE Dataset
    ohe = fe_pipeline.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
    cat_names = list(ohe.get_feature_names_out(all_cats))
    is_discrete = np.array([c in cat_names for c in Xt_tr.columns], dtype=bool)

    mi = mutual_info_classif(Xt_tr.values, y_train.values, discrete_features=is_discrete, random_state=SEED)
    mi_series = pd.Series(mi, index=Xt_tr.columns).sort_values(ascending=False)

    # Lưu biểu đồ Top MI
    topN = mi_series.head(min(20, len(mi_series))).iloc[::-1]
    plt.figure(figsize=(10, max(6, 0.35 * len(topN))))
    plt.barh(topN.index, topN.values)
    plt.title('Top MI scores (Train)')
    plt.xlabel('MI score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    plt.savefig('top_mi_scores.pdf', bbox_inches='tight')
    plt.close()

    # Chọn Top K features bằng MI
    K = raw_df.columns.drop('target').shape[0]  # Giữ số lượng feature bằng số cột gốc (13)
    topk_mi_cols = list(mi_series.head(K).index)

    fe_tr = Xt_tr[topk_mi_cols]
    fe_va = Xt_va[topk_mi_cols]
    fe_te = Xt_te[topk_mi_cols]
    save_dataset(fe_tr, y_train, fe_va, y_val, fe_te, y_test, 'fe')

    # 7. Decision Tree on Feature Engineering Dataset
    dt_fe_pipeline = Pipeline([
        ('preprocess', fe_pipeline),
        ('decision_tree', DecisionTreeClassifier(random_state=SEED))
    ])
    dt_fe_pipeline.fit(X_train, y_train)

    pipeline_feature_names = dt_fe_pipeline.named_steps['preprocess'].get_feature_names_out()
    fi_fe = pd.Series(
        dt_fe_pipeline.named_steps['decision_tree'].feature_importances_,
        index=pipeline_feature_names
    ).sort_values(ascending=False)

    selected_fe_features = fi_fe.head(K_FEATURES).index.tolist()
    print(f"\nTop {K_FEATURES} selected FE features by DT: {selected_fe_features}")

    X_fe_dt_train = Xt_tr[selected_fe_features]
    X_fe_dt_val = Xt_va[selected_fe_features]
    X_fe_dt_test = Xt_te[selected_fe_features]
    save_dataset(X_fe_dt_train, y_train, X_fe_dt_val, y_val, X_fe_dt_test, y_test, 'fe_dt')

    print("\nQuá trình tạo dataset hoàn tất. Các file CSV đã được lưu trong thư mục 'splits'.")

if __name__ == "__main__":
    main()