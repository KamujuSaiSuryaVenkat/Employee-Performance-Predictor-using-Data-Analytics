# src/data_generator.py

import numpy as np
import pandas as pd
from src.config import *
from src.utils import clip_values, add_noise

np.random.seed(RANDOM_SEED)


def generate_employee_data(num_employees=NUM_EMPLOYEES):
    data = {}

    # -----------------------------
    # DEMOGRAPHICS
    # -----------------------------
    data['age'] = np.random.randint(22, 60, num_employees)
    data['education_level'] = np.random.choice(EDUCATION_LEVELS, num_employees)
    data['experience_years'] = clip_values(
        data['age'] - np.random.randint(21, 25, num_employees), 0, 35
    )

    # -----------------------------
    # JOB CONTEXT
    # -----------------------------
    data['department'] = np.random.choice(DEPARTMENTS, num_employees)
    data['job_level'] = np.random.randint(1, 6, num_employees)
    data['manager_tenure'] = np.random.randint(0, 20, num_employees)

    # -----------------------------
    # PRODUCTIVITY
    # -----------------------------
    data['projects_completed'] = np.random.randint(1, 15, num_employees)
    data['story_points_completed'] = np.random.randint(10, 200, num_employees)
    data['billable_hours_ratio'] = np.random.uniform(0.3, 1.0, num_employees)

    # -----------------------------
    # WORK BEHAVIOR
    # -----------------------------
    data['avg_task_delay_days'] = np.random.uniform(-2, 20, num_employees)
    data['on_time_delivery_rate'] = np.random.uniform(0.4, 1.0, num_employees)

    # -----------------------------
    # QUALITY METRICS
    # -----------------------------
    data['bug_count'] = np.random.randint(0, 50, num_employees)
    data['code_review_score'] = np.random.uniform(1, 5, num_employees)
    data['defect_density'] = np.random.uniform(0, 10, num_employees)

    # -----------------------------
    # LEARNING
    # -----------------------------
    data['training_hours'] = np.random.randint(0, 100, num_employees)
    data['certifications_count'] = np.random.randint(0, 10, num_employees)

    # -----------------------------
    # ATTENDANCE
    # -----------------------------
    data['sick_days'] = np.random.randint(0, 15, num_employees)
    data['unplanned_leaves'] = np.random.randint(0, 10, num_employees)

    # -----------------------------
    # FEEDBACK
    # -----------------------------
    data['peer_feedback_score'] = np.random.uniform(1, 5, num_employees)
    data['manager_score'] = np.random.uniform(1, 5, num_employees)

    # -----------------------------
    # RECOGNITION
    # -----------------------------
    data['kudos_count'] = np.random.randint(0, 20, num_employees)
    data['promotions_last_2_years'] = np.random.randint(0, 3, num_employees)

    # -----------------------------
    # COMPENSATION
    # -----------------------------
    data['salary_percentile'] = np.random.uniform(0, 100, num_employees)

    df = pd.DataFrame(data)

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    df['norm_code_review'] = df['code_review_score'] / 5
    df['norm_peer_feedback'] = df['peer_feedback_score'] / 5
    df['norm_manager_score'] = df['manager_score'] / 5
    df['norm_training'] = df['training_hours'] / 100
    df['norm_certifications'] = df['certifications_count'] / 10
    df['norm_bug'] = df['bug_count'] / 50
    df['norm_delay'] = df['avg_task_delay_days'] / 20

    # -----------------------------
    # PERFORMANCE SCORE
    # -----------------------------
    performance_score = (
        0.20 * df['on_time_delivery_rate'] +
        0.15 * df['norm_code_review'] +
        0.10 * df['norm_peer_feedback'] +
        0.15 * df['norm_manager_score'] +
        0.10 * df['norm_training'] +
        0.05 * df['norm_certifications'] +
        0.10 * df['billable_hours_ratio'] -
        0.10 * df['norm_bug'] -
        0.05 * df['norm_delay']
    )

    # Add noise
    noise = add_noise(num_employees)
    df['performance_score'] = clip_values(performance_score + noise, 0, 1)

    # -----------------------------
    # PERFORMANCE BAND
    # -----------------------------
    conditions = [
        df['performance_score'] >= HIGH_PERFORMANCE_THRESHOLD,
        df['performance_score'] >= MEDIUM_PERFORMANCE_THRESHOLD
    ]

    choices = ['High', 'Medium']
    df['performance_band'] = np.select(conditions, choices, default='Low')

    # Drop intermediate columns
    df.drop(columns=[col for col in df.columns if 'norm_' in col], inplace=True)

    return df


def save_dataset(df, path=RAW_DATA_PATH):
    df.to_csv(path, index=False)
    print(f"Dataset saved at {path}")


if __name__ == "__main__":
    df = generate_employee_data()
    save_dataset(df)
    print(df.head())