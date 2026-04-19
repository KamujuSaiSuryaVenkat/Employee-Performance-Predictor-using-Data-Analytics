# main.py

from src.data_generator import generate_employee_data, save_dataset
from src.eda import generate_all_eda
from src.preprocessing import get_processed_data
from src.model import build_model, train_model, evaluate_model
from src.explainability import get_feature_importance, explain_employee, generate_recommendations


if __name__ == "__main__":

    # Step 1: Generate Data
    df = generate_employee_data()
    save_dataset(df)

    print("\nDataset Preview:\n")
    print(df.head())

    # Step 2: EDA
    generate_all_eda()

    # Step 3: Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = get_processed_data()

    # Step 4: Model
    model = build_model(preprocessor, y_train)
    model = train_model(model, X_train, y_train)

    # Step 5: Evaluation
    evaluate_model(model, X_test, y_test)

        # Step 6: Explainability
    top_features = get_feature_importance(model, X_test, y_test)

    # Step 7: Explain one employee
    sample = X_test.iloc[[0]]
    pred, reasons = explain_employee(model, sample, top_features)

    # Step 8: Recommendations
    generate_recommendations(reasons)