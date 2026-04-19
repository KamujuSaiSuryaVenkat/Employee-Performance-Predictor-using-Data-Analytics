# Employee Performance Predictor

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2200&pause=700&color=0A7E8C&center=true&vCenter=true&width=800&lines=AI-Powered+Employee+Performance+Prediction;Smart+Insights+for+HR+Decisions;Streamlit+App+and+ML+Pipeline" alt="Animated intro" />
</p>

<p align="center">💼 📊 ⚡</p>

A Data Science project that predicts employee performance bands (Low, Medium, High) using machine learning and provides explainable HR insights.

## Features

- Synthetic employee dataset generation
- Data preprocessing pipeline with imputation, scaling, and encoding
- Model training and evaluation for performance prediction
- Feature-importance-based explainability
- Streamlit UI for single and bulk predictions

## Project Structure

```text
Project-3/
├── app.py
├── main.py
├── requirements.txt
├── data/
│   └── raw/
├── outputs/
│   └── plots/
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_generator.py
    ├── eda.py
    ├── explainability.py
    ├── model.py
    ├── preprocessing.py
    └── utils.py
```

## Setup

1. Create a virtual environment.
2. Activate it.
3. Install dependencies.
4. Generate the dataset.
5. Run the Streamlit app.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```

## Usage

- Single Employee mode: Enter employee details and predict performance.
- Bulk Prediction mode: Upload a CSV and get predictions for all employees.

## Typical Model Insights

- Important features often include:
  - code_review_score
  - on_time_delivery_rate
  - manager_score
  - training_hours
  - bug_count
- HR recommendations are generated from top contributing factors.

## Troubleshooting

- If you get missing package errors, make sure app and pip use the same Python interpreter.
- If dataset file is missing, run:

```powershell
python main.py
```

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## License

This project is licensed under the MIT License. See LICENSE.md for details.

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=1800&pause=900&color=F97316&center=true&vCenter=true&width=620&lines=Thanks+for+checking+out+this+project!;If+you+like+it%2C+consider+starring+it!" alt="Animated footer" />
</p>

<p align="center">🎉 ⭐</p>
