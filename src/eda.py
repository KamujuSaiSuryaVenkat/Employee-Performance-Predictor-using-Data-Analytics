# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.config import RAW_DATA_PATH

OUTPUT_PATH = "outputs/plots"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_data():
    return pd.read_csv(RAW_DATA_PATH)


def dataset_overview(df):
    print("\n📊 DATASET OVERVIEW\n")
    print(df.info())
    print("\nSummary Statistics:\n")
    print(df.describe())


# 1️⃣ Performance Distribution
def performance_distribution(df):
    plt.figure()
    df['performance_band'].value_counts().plot(kind='bar')
    plt.title("Performance Band Distribution")
    plt.savefig(f"{OUTPUT_PATH}/performance_distribution.png")
    plt.close()


# 2️⃣ Score Distribution
def performance_score_distribution(df):
    plt.figure()
    df['performance_score'].hist()
    plt.title("Performance Score Distribution")
    plt.savefig(f"{OUTPUT_PATH}/performance_score_distribution.png")
    plt.close()


# 3️⃣ Correlation Heatmap
def correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr)
    plt.title("Correlation Heatmap")
    plt.savefig(f"{OUTPUT_PATH}/correlation_heatmap.png")
    plt.close()


# 4️⃣ Key Feature Analysis
def key_feature_analysis(df):
    features = [
        'on_time_delivery_rate',
        'bug_count',
        'manager_score',
        'training_hours'
    ]

    for feature in features:
        plt.figure()
        sns.boxplot(x='performance_band', y=feature, data=df)
        plt.title(f"{feature} vs Performance")
        plt.savefig(f"{OUTPUT_PATH}/{feature}_vs_performance.png")
        plt.close()


# 5️⃣ Department Insight
def department_analysis(df):
    plt.figure()
    sns.countplot(x='department', hue='performance_band', data=df)
    plt.title("Department vs Performance")
    plt.xticks(rotation=30)
    plt.savefig(f"{OUTPUT_PATH}/department_vs_performance.png")
    plt.close()


def generate_all_eda():
    df = load_data()

    dataset_overview(df)

    performance_distribution(df)
    performance_score_distribution(df)
    correlation_heatmap(df)
    key_feature_analysis(df)
    department_analysis(df)

    print("\n✅ CLEAN EDA Completed (only important plots)")


if __name__ == "__main__":
    generate_all_eda()