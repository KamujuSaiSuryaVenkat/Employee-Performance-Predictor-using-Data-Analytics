# src/explainability.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


OUTPUT_PATH = "outputs/plots"


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
def get_feature_importance(model, X_test, y_test):
    print("\n🔍 Computing Feature Importance...\n")

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        scoring="f1_macro"
    )

    # ✅ FIX: use original feature names
    feature_names = X_test.columns

    importances = pd.Series(result.importances_mean, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(10)

    print("Top Features:\n", top_features)

    # Plot
    plt.figure()
    top_features.sort_values().plot(kind='barh')
    plt.title("Top Feature Importance")
    plt.savefig(f"{OUTPUT_PATH}/feature_importance.png")
    plt.close()

    return top_features


# -----------------------------
# EMPLOYEE LEVEL EXPLANATION
# -----------------------------
def explain_employee(model, X_sample, top_features):
    prediction = model.predict(X_sample)[0]

    print("\n🧠 Employee Prediction:", prediction)

    explanations = []

    # Get top important features only
    important_features = top_features.index.tolist()

    for feature in important_features[:5]:  # top 5 drivers
        value = X_sample[feature].values[0]

        if feature == "on_time_delivery_rate" and value < 0.6:
            explanations.append(f"Low {feature}")

        elif feature == "bug_count" and value > 30:
            explanations.append(f"High {feature}")

        elif feature == "manager_score" and value < 3:
            explanations.append(f"Low {feature}")

        elif feature == "training_hours" and value < 20:
            explanations.append(f"Low {feature}")

        elif feature == "code_review_score" and value < 3:
            explanations.append(f"Low {feature}")

    print("Top Contributing Factors:")
    for e in explanations:
        print("-", e)

    return prediction, explanations

# -----------------------------
# HR RECOMMENDATION ENGINE
# -----------------------------
def generate_recommendations(explanations):
    print("\n📌 HR Recommendations:\n")

    for e in explanations:
        if "delivery" in e:
            print("- Assign time management training")

        elif "bug" in e:
            print("- Provide code quality training")

        elif "manager" in e:
            print("- Schedule manager feedback sessions")

        elif "training" in e:
            print("- Enroll in upskilling programs")