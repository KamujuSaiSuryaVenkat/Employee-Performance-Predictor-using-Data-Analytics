# Employee Performance Predictor 🚀

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=2200&pause=700&color=0A7E8C&center=true&vCenter=true&width=800&lines=💼+AI-Powered+Employee+Performance+Prediction;📊+Smart+Insights+for+HR+Decisions;⚡+Streamlit+App+%2B+ML+Pipeline" alt="Animated intro" />
</p>

A practical Data Science project that predicts employee performance bands (`Low`, `Medium`, `High`) using machine learning, and provides explainable insights and HR-friendly recommendations.

## Highlights ✨

- 📦 Synthetic employee data generation with realistic workforce features.
- 🧹 Clean preprocessing pipeline with imputation, scaling, and encoding.
- 🤖 Classification model training and evaluation.
- 🔍 Feature-importance-based explainability.
- 🖥️ Streamlit UI for single and bulk predictions.

## Project Structure 🗂️

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
    ├── config.py
    ├── data_generator.py
    ├── eda.py
    ├── explainability.py
    ├── model.py
    ├── preprocessing.py
    └── utils.py
```

## Quick Start ⚙️

1. Create and activate a virtual environment.
2. Install dependencies.
3. Generate dataset (if needed).
4. Run the Streamlit app.

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```

## Example Outputs 📈

- Model metrics: precision, recall, F1-score, confusion matrix.
- Top drivers: `code_review_score`, `on_time_delivery_rate`, `manager_score`, etc.
- Actionable HR recommendations based on predicted performance factors.

## Tech Stack 🧠

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit

## Notes 📝

- Ensure you run the app using the same interpreter where dependencies are installed.
- Dataset path handling is project-root based for better reliability.

## Author 👨‍💻

Created for IITD Data Science Project-3.

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=18&duration=1800&pause=900&color=F97316&center=true&vCenter=true&width=600&lines=🎉+Thanks+for+checking+out+this+project!;⭐+If+you+like+it%2C+consider+starring+it!" alt="Animated footer" />
</p>
