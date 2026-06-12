import os
import gc
import numpy as np
import pandas as pd
import torch
import optuna
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SYSTEM] Using device for NLP extraction: {device}")

# Global Config
CONFIG = {
    "train_path": "train.csv",
    "test_path": "test_x.csv",
    "submission_path": "submission.csv",
    "bert_model_name": "dbmdz/bert-base-turkish-cased",
    "nlp_pca_components": 15,
    "batch_size": 128,
    "n_folds": 5,
    "optuna_trials": 50,
    "random_state": 42
}

def clean_and_preprocess(df, is_train=True):
    print(f"[PREPROCESS] Cleaning data (is_train={is_train})...")
    df = df.copy()
    
    # Missing Value Flags & Imputation
    # For numeric columns with missing values:
    missing_cols = [
        "english_exam_score", "portfolio_score", "github_avg_stars", 
        "open_source_contribution_count", "linkedin_profile_score", "hr_interview_score"
    ]
    
    # We also handle internship_duration_months specially: if internship_count is 0 or it's nan, make it 0.
    df["internship_duration_months_is_missing"] = df["internship_duration_months"].isnull().astype(int)
    df["internship_duration_months"] = df["internship_duration_months"].fillna(0.0)
    
    for col in missing_cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isnull().astype(int)
            # Impute with median
            if is_train:
                median_val = df[col].median()
                # Store median for test set (we hardcode/use train stats)
                # To be simple and robust, let's define median values based on train data stats
            # Since we have the whole dataframe here, we can compute medians directly
            # For test set, we will use median of the column in the current dataframe (which is extremely close)
            df[col] = df[col].fillna(df[col].median())
            
    return df

def feature_engineering(df):
    print("[FEATURES] Engineering new features...")
    df = df.copy()
    
    # --- Zaman & Demografi ---
    df["years_since_graduation"] = df["application_year"] - df["graduation_year"]
    df["graduation_age"] = df["graduation_year"] - (df["application_year"] - df["age"])
    
    # --- Akademik Disiplin Skoru ---
    # Low attendance rate and high failed courses count are penalized
    df["academic_discipline_score"] = (df["cgpa"] * df["attendance_rate"] / 100.0) - (df["failed_courses_count"] * 0.2)
    
    # --- Teknik Beceri Gruplama & İstatistikleri ---
    tech_score_cols = [
        "coding_score", "problem_solving_score", "data_structures_score", 
        "sql_score", "machine_learning_score", "backend_score", 
        "frontend_score", "cloud_score", "devops_score"
    ]
    df["software_eng_score"] = df[["backend_score", "frontend_score", "devops_score"]].mean(axis=1)
    df["data_ai_score"] = df[["sql_score", "machine_learning_score", "data_structures_score"]].mean(axis=1)
    
    df["total_tech_score_mean"] = df[tech_score_cols].mean(axis=1)
    df["total_tech_score_std"] = df[tech_score_cols].std(axis=1)
    df["total_tech_score_max"] = df[tech_score_cols].max(axis=1)
    df["total_tech_score_min"] = df[tech_score_cols].min(axis=1)
    
    # Specialization difference (positive: AI/Data oriented, negative: SE oriented)
    df["specialization_flag"] = df["data_ai_score"] - df["software_eng_score"]
    
    # --- Pratik ve Saha Deneyimi ---
    df["total_field_experience_months"] = df["internship_duration_months"] + (df["freelance_project_count"] * 3.0)
    df["github_impact_score"] = df["github_repo_count"] * df["github_avg_stars"]
    df["total_project_count"] = df["real_client_project_count"] + df["freelance_project_count"]
    df["hackathon_efficiency"] = df["hackathon_awards"] / (df["hackathon_count"] + 1.0)
    
    # --- CV & Mülakat ---
    df["cv_conversion_rate"] = df["interviews_attended"] / (df["applications_sent"] + 1.0)
    df["interview_total_score_mean"] = df[["technical_interview_score", "hr_interview_score"]].mean(axis=1)
    
    # --- Sosyal Beceriler ---
    social_cols = ["communication_score", "teamwork_score", "leadership_score", "presentation_score"]
    df["soft_skills_mean"] = df[social_cols].mean(axis=1)
    df["soft_skills_std"] = df[social_cols].std(axis=1)
    df["soft_skills_max"] = df[social_cols].max(axis=1)
    df["soft_skills_min"] = df[social_cols].min(axis=1)
    
    # --- Etkileşim Özellikleri ---
    df["cgpa_x_coding"] = df["cgpa"] * df["coding_score"]
    df["experience_x_project_quality"] = df["total_field_experience_months"] * df["project_quality_score"]
    df["portfolio_x_github"] = df["portfolio_score"] * df["github_impact_score"]
    
    # --- Türkçe Mentor Değerlendirmesi Sentiment Analizi (Kural Tabanlı) ---
    positive_words = [
        "başarı", "iyi", "harika", "mükemmel", "güçlü", "başarılı", "yetkinlik", "tebrikler", 
        "potansiyel", "olumlu", "özveri", "aktif", "avantaj", "güzel", "üst düzey", "takdir", 
        "motivasyon", "uyum", "örnek", "azim", "katkı", "donanımlı", "olağanüstü", "uzman"
    ]
    negative_words = [
        "eksik", "zayıf", "yetersiz", "geliştirmesi gerekiyor", "çalışması gerekiyor", "düşük", 
        "zorluk", "sınırlı", "odaklanmalı", "destek alması", "gelişime açık", "ihtiyacı var"
    ]
    
    def calculate_sentiment(text):
        if not isinstance(text, str):
            return 0.0, 0, 0
        text_lower = text.lower()
        pos_count = sum(text_lower.count(word) for word in positive_words)
        neg_count = sum(text_lower.count(word) for word in negative_words)
        
        # Length features
        char_len = len(text)
        word_count = len(text_lower.split())
        
        sentiment_score = (pos_count - neg_count) / (pos_count + neg_count + 1.0)
        return sentiment_score, char_len, word_count
    
    sent_results = [calculate_sentiment(t) for t in df["mentor_feedback_text"]]
    df["mentor_sentiment_score"] = [r[0] for r in sent_results]
    df["mentor_char_length"] = [r[1] for r in sent_results]
    df["mentor_word_count"] = [r[2] for r in sent_results]
    
    return df

def extract_nlp_features(df_train, df_test):
    print("[NLP] Loading BERT model and tokenizer for embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["bert_model_name"])
    model = AutoModel.from_pretrained(CONFIG["bert_model_name"]).to(device)
    model.eval()
    
    def get_embeddings(texts):
        embeddings = []
        # Process in batches
        for i in tqdm(range(0, len(texts), CONFIG["batch_size"]), desc="Extracting embeddings"):
            batch_texts = texts[i:i + CONFIG["batch_size"]]
            # Tokenize
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Extract CLS token embeddings (first token in output sequence)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_emb)
        return np.concatenate(embeddings, axis=0)
    
    print("[NLP] Extracting training text embeddings...")
    train_texts = df_train["mentor_feedback_text"].fillna("").tolist()
    train_embs = get_embeddings(train_texts)
    
    print("[NLP] Extracting testing text embeddings...")
    test_texts = df_test["mentor_feedback_text"].fillna("").tolist()
    test_embs = get_embeddings(test_texts)
    
    # Fit PCA on training embeddings and project both train and test
    print(f"[NLP] Squeezing embeddings with PCA from 768 down to {CONFIG['nlp_pca_components']} components...")
    pca = PCA(n_components=CONFIG["nlp_pca_components"], random_state=CONFIG["random_state"])
    train_pca = pca.fit_transform(train_embs)
    test_pca = pca.transform(test_embs)
    
    # Create column names
    pca_cols = [f"nlp_pca_{i}" for i in range(CONFIG["nlp_pca_components"])]
    
    # Create DataFrames
    df_train_pca = pd.DataFrame(train_pca, columns=pca_cols, index=df_train.index)
    df_test_pca = pd.DataFrame(test_pca, columns=pca_cols, index=df_test.index)
    
    # Join back to original dfs
    df_train = pd.concat([df_train, df_train_pca], axis=1)
    df_test = pd.concat([df_test, df_test_pca], axis=1)
    
    return df_train, df_test

def target_encode(train_df, val_df, test_df, cat_cols, target_col, smoothing=10):
    """
    Computes smoothed target encoding inside cross validation folds to prevent data leakage.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    global_mean = train_df[target_col].mean()
    
    for col in cat_cols:
        # Group by and aggregate sum and count
        stats = train_df.groupby(col)[target_col].agg(['count', 'mean'])
        
        # Smoothed formula: (count * mean + smoothing * global_mean) / (count + smoothing)
        smooth_val = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        # Map back to train, val and test
        train_df[f"{col}_te"] = train_df[col].map(smooth_val).fillna(global_mean).astype(float)
        val_df[f"{col}_te"] = val_df[col].map(smooth_val).fillna(global_mean).astype(float)
        test_df[f"{col}_te"] = test_df[col].map(smooth_val).fillna(global_mean).astype(float)
        
    return train_df, val_df, test_df

def main():
    print("[SYSTEM] Starting BTK Datathon 2026 Pipeline...")
    
    # 1. Load Data
    train_df = pd.read_csv(CONFIG["train_path"])
    test_df = pd.read_csv(CONFIG["test_path"])
    
    # Preprocessing
    train_df = clean_and_preprocess(train_df, is_train=True)
    test_df = clean_and_preprocess(test_df, is_train=False)
    
    # Feature Engineering
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # NLP extraction (BERT + PCA)
    train_df, test_df = extract_nlp_features(train_df, test_df)
    
    # Identify column types
    id_col = "student_id"
    target_col = "career_success_score"
    text_col = "mentor_feedback_text"
    
    cat_cols = ["department", "university_tier", "target_role", "hobby", "preferred_social_media_platform"]
    
    # Ensure cat_cols are categorical type
    for col in cat_cols:
        train_df[col] = train_df[col].astype("category")
        test_df[col] = test_df[col].astype("category")
        
    # Label encode categorical columns for models that need integer labels
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined values to avoid unseen label errors
        combined = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined)
        train_df[f"{col}_le"] = le.transform(train_df[col].astype(str))
        test_df[f"{col}_le"] = le.transform(test_df[col].astype(str))
        le_dict[col] = le
        
    # Drop raw id, target, and text columns from training features list
    feature_cols = [c for c in train_df.columns if c not in [id_col, target_col, text_col]]
    
    print(f"[SYSTEM] Total engineered features: {len(feature_cols)}")
    
    # Prepare cross-validation splits
    kf = KFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_state"])
    
    # Out of fold predictions
    oof_xgb = np.zeros(len(train_df))
    oof_lgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    
    # Test predictions (we will average predictions across all folds)
    test_preds_xgb = np.zeros(len(test_df))
    test_preds_lgb = np.zeros(len(test_df))
    test_preds_cat = np.zeros(len(test_df))
    
    # Let's perform target encoding for K-Fold splits.
    # We will pre-generate target-encoded values fold by fold.
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    
    # Initialize TE columns in encoded datasets
    for col in cat_cols:
        train_df_encoded[f"{col}_te"] = np.nan
        test_df_encoded[f"{col}_te"] = np.zeros(len(test_df))
        
    # Inside folds, we compute target encoding and place them back
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]
        
        # Apply target encoding
        t_fold_train, t_fold_val, t_test = target_encode(
            fold_train, fold_val, test_df, cat_cols, target_col, smoothing=10
        )
        
        # Save to encoded dataframes
        for col in cat_cols:
            train_df_encoded.iloc[val_idx, train_df_encoded.columns.get_loc(f"{col}_te")] = t_fold_val[f"{col}_te"]
            # Accumulate test TE values divided by fold count
            test_df_encoded[f"{col}_te"] += t_test[f"{col}_te"] / CONFIG["n_folds"]
            
        # For the train parts, we can use the self-encoded values as well
        # In actual training, we target encode the train fold inside the fold loop.
        
    # XGBoost and LightGBM will use label encoded + target encoded features + numeric features.
    # CatBoost can use either, or native categoricals. Let's provide native categoricals to CatBoost, and target + label encoded ones to LightGBM/XGBoost.
    
    # Build feature sets
    # We will exclude raw string/category features for XGBoost & LightGBM
    exclude_raw_cat = cat_cols
    xgb_features = [c for c in train_df_encoded.columns if c not in [id_col, target_col, text_col] + exclude_raw_cat]
    lgb_features = xgb_features.copy()
    
    # CatBoost features will include raw categoricals as well
    cat_features = cat_cols.copy()
    catboost_features = [c for c in train_df.columns if c not in [id_col, target_col, text_col]]
    
    # ------------------ OPTUNA STUDY FOR HYPERPARAMETERS ------------------
    print("\n" + "="*50)
    print("[OPTUNA] Tuning XGBoost...")
    print("="*50)
    
    def objective_xgb(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": CONFIG["random_state"],
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "tree_method": "hist"
        }
        
        # We do a fast 3-fold CV inside Optuna to speed up trials
        kf_fast = KFold(n_splits=3, shuffle=True, random_state=CONFIG["random_state"])
        rmses = []
        
        for tr_idx, va_idx in kf_fast.split(train_df):
            X_tr, y_tr = train_df_encoded.iloc[tr_idx][xgb_features], train_df_encoded.iloc[tr_idx][target_col]
            X_va, y_va = train_df_encoded.iloc[va_idx][xgb_features], train_df_encoded.iloc[va_idx][target_col]
            
            # Since Optuna does fast CV, we target-encode fold-specifically inside the objective function
            t_tr, t_va, _ = target_encode(train_df.iloc[tr_idx], train_df.iloc[va_idx], test_df, cat_cols, target_col)
            X_tr = t_tr[xgb_features]
            X_va = t_va[xgb_features]
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
            preds = model.predict(X_va)
            rmse = root_mean_squared_error(y_va, preds)
            rmses.append(rmse)
            
        return np.mean(rmses)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(objective_xgb, n_trials=CONFIG["optuna_trials"])
    best_xgb_params = study_xgb.best_params
    print(f"[OPTUNA] Best XGBoost Params: {best_xgb_params}")
    print(f"[OPTUNA] Best XGBoost CV Score: {study_xgb.best_value:.4f}")
    
    print("\n" + "="*50)
    print("[OPTUNA] Tuning LightGBM...")
    print("="*50)
    
    def objective_lgb(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": CONFIG["random_state"],
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "n_jobs": -1
        }
        
        kf_fast = KFold(n_splits=3, shuffle=True, random_state=CONFIG["random_state"])
        rmses = []
        
        for tr_idx, va_idx in kf_fast.split(train_df):
            t_tr, t_va, _ = target_encode(train_df.iloc[tr_idx], train_df.iloc[va_idx], test_df, cat_cols, target_col)
            X_tr, y_tr = t_tr[lgb_features], train_df.iloc[tr_idx][target_col]
            X_va, y_va = t_va[lgb_features], train_df.iloc[va_idx][target_col]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)]
            )
            preds = model.predict(X_va)
            rmse = root_mean_squared_error(y_va, preds)
            rmses.append(rmse)
            
        return np.mean(rmses)

    study_lgb = optuna.create_study(direction="minimize")
    study_lgb.optimize(objective_lgb, n_trials=CONFIG["optuna_trials"])
    best_lgb_params = study_lgb.best_params
    print(f"[OPTUNA] Best LightGBM Params: {best_lgb_params}")
    print(f"[OPTUNA] Best LightGBM CV Score: {study_lgb.best_value:.4f}")

    print("\n" + "="*50)
    print("[OPTUNA] Tuning CatBoost...")
    print("="*50)
    
    def objective_cat(trial):
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": CONFIG["random_state"],
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "iterations": trial.suggest_int("iterations", 300, 1500),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "task_type": "GPU" if torch.cuda.is_available() else "CPU",
            "verbose": 0
        }
        
        kf_fast = KFold(n_splits=3, shuffle=True, random_state=CONFIG["random_state"])
        rmses = []
        
        for tr_idx, va_idx in kf_fast.split(train_df):
            X_tr, y_tr = train_df.iloc[tr_idx][catboost_features], train_df.iloc[tr_idx][target_col]
            X_va, y_va = train_df.iloc[va_idx][catboost_features], train_df.iloc[va_idx][target_col]
            
            # CatBoost handles raw categoricals natively, but they must be cast to string/categorical
            # We copy X_tr and X_va to make sure categorical types are correctly handled
            X_tr_cb = X_tr.copy()
            X_va_cb = X_va.copy()
            for col in cat_features:
                X_tr_cb[col] = X_tr_cb[col].astype(str)
                X_va_cb[col] = X_va_cb[col].astype(str)
                
            train_pool = Pool(X_tr_cb, y_tr, cat_features=cat_features)
            val_pool = Pool(X_va_cb, y_va, cat_features=cat_features)
            
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool)
            preds = model.predict(val_pool)
            rmse = root_mean_squared_error(y_va, preds)
            rmses.append(rmse)
            
        return np.mean(rmses)

    study_cat = optuna.create_study(direction="minimize")
    study_cat.optimize(objective_cat, n_trials=CONFIG["optuna_trials"])
    best_cat_params = study_cat.best_params
    print(f"[OPTUNA] Best CatBoost Params: {best_cat_params}")
    print(f"[OPTUNA] Best CatBoost CV Score: {study_cat.best_value:.4f}")

    # ------------------ FINAL 5-FOLD CV AND TRAINING ------------------
    print("\n" + "="*50)
    print("[FINAL] Starting 5-Fold training with best hyperparameters...")
    print("="*50)
    
    # Configure final models with optimal parameters
    xgb_base_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": CONFIG["random_state"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tree_method": "hist"
    }
    xgb_base_params.update(best_xgb_params)
    
    lgb_base_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "random_state": CONFIG["random_state"],
        "n_jobs": -1
    }
    lgb_base_params.update(best_lgb_params)
    
    cat_base_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": CONFIG["random_state"],
        "task_type": "GPU" if torch.cuda.is_available() else "CPU",
        "verbose": 0
    }
    cat_base_params.update(best_cat_params)
    
    # 5-Fold Training
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"\n--- Fold {fold+1} / {CONFIG['n_folds']} ---")
        
        # Apply fold-specific target encoding
        t_train, t_val, t_test = target_encode(
            train_df.iloc[train_idx], train_df.iloc[val_idx], test_df, cat_cols, target_col
        )
        
        # Prepare datasets
        X_tr_xgb, y_tr = t_train[xgb_features], train_df.iloc[train_idx][target_col]
        X_va_xgb, y_va = t_val[xgb_features], train_df.iloc[val_idx][target_col]
        X_te_xgb = t_test[xgb_features]
        
        X_tr_lgb = X_tr_xgb.copy()
        X_va_lgb = X_va_xgb.copy()
        X_te_lgb = X_te_xgb.copy()
        
        # --- XGBoost ---
        print("[Fold] Training XGBoost...")
        model_xgb = xgb.XGBRegressor(**xgb_base_params)
        model_xgb.fit(
            X_tr_xgb, y_tr,
            eval_set=[(X_va_xgb, y_va)],
            verbose=False
        )
        oof_xgb[val_idx] = model_xgb.predict(X_va_xgb)
        test_preds_xgb += model_xgb.predict(X_te_xgb) / CONFIG["n_folds"]
        print(f"       XGB Fold RMSE: {root_mean_squared_error(y_va, oof_xgb[val_idx]):.4f}")
        
        # --- LightGBM ---
        print("[Fold] Training LightGBM...")
        model_lgb = lgb.LGBMRegressor(**lgb_base_params)
        model_lgb.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_va_lgb, y_va)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_va_lgb)
        test_preds_lgb += model_lgb.predict(X_te_lgb) / CONFIG["n_folds"]
        print(f"       LGBM Fold RMSE: {root_mean_squared_error(y_va, oof_lgb[val_idx]):.4f}")
        
        # --- CatBoost ---
        print("[Fold] Training CatBoost...")
        # Prepare native categoricals for CatBoost
        X_tr_cat = train_df.iloc[train_idx][catboost_features].copy()
        X_va_cat = train_df.iloc[val_idx][catboost_features].copy()
        X_te_cat = test_df[catboost_features].copy()
        
        for col in cat_features:
            X_tr_cat[col] = X_tr_cat[col].astype(str)
            X_va_cat[col] = X_va_cat[col].astype(str)
            X_te_cat[col] = X_te_cat[col].astype(str)
            
        train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va_cat, y_va, cat_features=cat_features)
        test_pool = Pool(X_te_cat, cat_features=cat_features)
        
        model_cat = CatBoostRegressor(**cat_base_params)
        model_cat.fit(train_pool, eval_set=val_pool)
        oof_cat[val_idx] = model_cat.predict(val_pool)
        test_preds_cat += model_cat.predict(test_pool) / CONFIG["n_folds"]
        print(f"       CatBoost Fold RMSE: {root_mean_squared_error(y_va, oof_cat[val_idx]):.4f}")
        
    # CV Score Reports
    y_true = train_df[target_col].values
    rmse_xgb = root_mean_squared_error(y_true, oof_xgb)
    rmse_lgb = root_mean_squared_error(y_true, oof_lgb)
    rmse_cat = root_mean_squared_error(y_true, oof_cat)
    
    print("\n" + "="*50)
    print("CROSS VALIDATION RESULTS")
    print("="*50)
    print(f"XGBoost CV RMSE:  {rmse_xgb:.4f}")
    print(f"LightGBM CV RMSE: {rmse_lgb:.4f}")
    print(f"CatBoost CV RMSE: {rmse_cat:.4f}")
    
    # ------------------ ENSEMBLE WEIGHT OPTIMIZATION ------------------
    print("\n" + "="*50)
    print("[ENSEMBLE] Finding optimal weights...")
    print("="*50)
    
    def loss_func(weights):
        w1, w2, w3 = weights
        pred = w1 * oof_xgb + w2 * oof_lgb + w3 * oof_cat
        return root_mean_squared_error(y_true, pred)
    
    # Constrain weights to sum to 1.0 and be non-negative
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    bounds = [(0.0, 1.0)] * 3
    initial_weights = [1/3, 1/3, 1/3]
    
    res = minimize(loss_func, initial_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    opt_w = res.x
    print(f"[ENSEMBLE] Optimal weights: XGBoost={opt_w[0]:.4f}, LightGBM={opt_w[1]:.4f}, CatBoost={opt_w[2]:.4f}")
    
    final_oof_pred = opt_w[0] * oof_xgb + opt_w[1] * oof_lgb + opt_w[2] * oof_cat
    final_cv_rmse = root_mean_squared_error(y_true, final_oof_pred)
    print(f"[ENSEMBLE] Final Ensemble CV RMSE: {final_cv_rmse:.4f}")
    
    # ------------------ PREDICT AND GENERATE SUBMISSION ------------------
    print("\n" + "="*50)
    print("[SUBMISSION] Generating predictions...")
    print("="*50)
    
    final_test_preds = opt_w[0] * test_preds_xgb + opt_w[1] * test_preds_lgb + opt_w[2] * test_preds_cat
    
    # Clip predictions between 0 and 100
    final_test_preds = np.clip(final_test_preds, 0.0, 100.0)
    
    # Create submission file
    submission = pd.DataFrame({
        "student_id": test_df[id_col],
        "career_success_score": final_test_preds
    })
    
    submission.to_csv(CONFIG["submission_path"], index=False)
    print(f"[SUBMISSION] Saved successfully to {CONFIG['submission_path']}!")
    print(f"[SUBMISSION] Shape: {submission.shape}")
    print("\nPreview of predictions:")
    print(submission.head(10))

if __name__ == "__main__":
    main()
