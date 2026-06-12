import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def add_features(df):
    """
    Perform professional, leak-free feature engineering for students' success prediction.
    """
    df = df.copy()
    
    # 1. Technical Skill Features
    tech_cols = [
        'coding_score', 'problem_solving_score', 'data_structures_score',
        'sql_score', 'machine_learning_score', 'backend_score',
        'frontend_score', 'cloud_score', 'devops_score'
    ]
    # Check if cols exist before operations
    existing_tech = [c for c in tech_cols if c in df.columns]
    if existing_tech:
        df['avg_tech_score'] = df[existing_tech].mean(axis=1)
        df['max_tech_score'] = df[existing_tech].max(axis=1)
        df['min_tech_score'] = df[existing_tech].min(axis=1)
        df['std_tech_score'] = df[existing_tech].std(axis=1)
        # Ratio of top scoring areas
        df['num_high_tech_scores'] = (df[existing_tech] >= 80).sum(axis=1)
        df['num_low_tech_scores'] = (df[existing_tech] < 50).sum(axis=1)

    # 2. Interview Performance
    interview_cols = ['technical_interview_score', 'hr_interview_score']
    existing_interview = [c for c in interview_cols if c in df.columns]
    if existing_interview:
        df['avg_interview_score'] = df[existing_interview].mean(axis=1)
        if len(existing_interview) == 2:
            df['interview_score_diff'] = df['technical_interview_score'] - df['hr_interview_score']
            
    # 3. Soft Skills
    social_cols = ['communication_score', 'teamwork_score', 'leadership_score', 'presentation_score']
    existing_social = [c for c in social_cols if c in df.columns]
    if existing_social:
        df['avg_social_score'] = df[existing_social].mean(axis=1)
        df['max_social_score'] = df[existing_social].max(axis=1)
        df['std_social_score'] = df[existing_social].std(axis=1)

    # 4. Academic Performance vs Attendance
    if 'cgpa' in df.columns and 'attendance_rate' in df.columns:
        df['academic_perf_index'] = df['cgpa'] * df['attendance_rate']
    
    # 5. Practical Experience and GitHub Activity
    if 'internship_count' in df.columns and 'real_client_project_count' in df.columns and 'freelance_project_count' in df.columns:
        df['total_practical_experience'] = df['internship_count'] + df['real_client_project_count'] + df['freelance_project_count']
        
    if 'github_repo_count' in df.columns and 'github_avg_stars' in df.columns:
        df['github_impact'] = df['github_repo_count'] * (df['github_avg_stars'] + 1.0)
        
    if 'hackathon_count' in df.columns and 'hackathon_awards' in df.columns:
        df['hackathon_award_ratio'] = df['hackathon_awards'] / (df['hackathon_count'] + 1.0)

    # 6. Applications and Interview Conversion
    if 'interviews_attended' in df.columns and 'applications_sent' in df.columns:
        df['interview_conversion_rate'] = df['interviews_attended'] / (df['applications_sent'] + 1.0)

    # 7. Portfolio & Profile Scores
    profile_cols = ['portfolio_score', 'linkedin_profile_score', 'cv_quality_score']
    existing_profiles = [c for c in profile_cols if c in df.columns]
    if existing_profiles:
        df['profile_strength_score'] = df[existing_profiles].mean(axis=1)

    return df

def main():
    print("[SYSTEM] Starting AutoGluon Regression Pipeline...")
    
    # 1. Load Data
    train_path = "train.csv"
    test_path = "test_x.csv"
    submission_path = "submission_autogluon.csv"
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Exclude mentor_feedback_text as requested by user
    exclude_cols = ["mentor_feedback_text"]
    print(f"[PREPROCESS] Excluding columns: {exclude_cols}")
    train_df = train_df.drop(columns=exclude_cols, errors="ignore")
    test_df = test_df.drop(columns=exclude_cols, errors="ignore")
    
    # 3. Apply Professional Feature Engineering
    print("[PREPROCESS] Applying custom feature engineering...")
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    
    # 4. Handle ID column & target column
    id_col = "student_id"
    target_col = "career_success_score"
    
    train_data = train_df.drop(columns=[id_col], errors="ignore")
    test_data = test_df.drop(columns=[id_col], errors="ignore")
    
    print(f"[DATA] Train shape (after feature engineering, without text & id): {train_data.shape}")
    print(f"[DATA] Test shape (after feature engineering, without text & id):  {test_data.shape}")
    
    # 5. Initialize TabularPredictor
    predictor = TabularPredictor(
        label=target_col,
        eval_metric="rmse",
        problem_type="regression",
        path="AutogluonModels/"
    )
    
    # 6. Fit TabularPredictor with Stack/Bagging and GPU
    print("\n" + "="*50)
    print("[TRAIN] Starting AutoGluon training on GPU (best_quality)...")
    print("="*50)
    
    predictor.fit(
        train_data=train_data,
        time_limit=3600,  # 1 hour limit
        presets="best_quality",
        num_gpus=1,
        auto_stack=True,
        ag_args_ensemble={'fold_fitting_strategy': 'sequential_local'},
        verbosity=2
    )
    
    print("\n" + "="*50)
    print("[LEADERBOARD] Training finished. Generating model leaderboard...")
    print("="*50)
    
    # Generate and print detailed leaderboard of all trained models
    leaderboard = predictor.leaderboard(extra_info=True)
    print(leaderboard)
    
    # Save leaderboard to text file for review
    leaderboard.to_csv("autogluon_leaderboard.csv", index=False)
    print("[SYSTEM] Leaderboard saved to autogluon_leaderboard.csv")
    
    # 7. Predict on Test Set
    print("\n" + "="*50)
    print("[PREDICT] Predicting on test set...")
    print("="*50)
    
    test_preds = predictor.predict(test_data)
    
    # Professional clipping to target limits [0.0, 100.0]
    test_preds_clipped = np.clip(test_preds, 0.0, 100.0)
    
    # 8. Generate Submission file
    submission = pd.DataFrame({
        "student_id": test_df[id_col],
        "career_success_score": test_preds_clipped
    })
    
    submission.to_csv(submission_path, index=False)
    print(f"[SYSTEM] Submission saved successfully to: {submission_path}")
    print(f"[SYSTEM] Submission shape: {submission.shape}")
    print("\nSubmission preview:")
    print(submission.head(10))

if __name__ == "__main__":
    main()
