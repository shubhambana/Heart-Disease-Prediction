# Heart Disease Risk Prediction System

A machine learning project that estimates the probability of heart disease from a patient's clinical parameters, served through an interactive Streamlit web app and a FastAPI REST backend.

🔗 **Live Demo:** [Add your Streamlit app link here]
📓 **Notebook:** [Disease_Prediction.ipynb](./notebook/Disease_Prediction.ipynb)

> **Disclaimer:** This is an educational portfolio project. It is not a medical device and must not be used for real diagnosis or treatment decisions.

---

## Overview

This project builds an end-to-end classification pipeline that predicts the likelihood of heart disease using key medical attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, and maximum heart rate achieved.

**Dataset:** Heart Disease Dataset (Kaggle / UCI)

**Approach:**
1. Cleaned and preprocessed the medical attributes and performed exploratory data analysis.
2. Compared three algorithms: Logistic Regression, Random Forest, and Support Vector Machine (SVM).
3. Selected **Random Forest** as the final model, which achieved the highest accuracy (**98.54%**).
4. Deployed the model through a Streamlit web app and a FastAPI REST endpoint.

---

## Key Features

- **Clinical parameter inputs:** interactive form for medical indicators (age, sex, chest pain type, ST depression, thalassemia, etc.).
- **Real-time risk score:** returns a heart disease probability for the entered patient details.
- **FastAPI backend:** REST endpoint for predictions from other applications.
- **Clean web interface:** simple Streamlit dashboard.

---

## Model Performance

| Model                        | Accuracy   |
|------------------------------|-----------:|
| Logistic Regression          | 79.51%     |
| **Random Forest** (final)    | **98.54%** |
| Support Vector Machine (SVM) | 88.78%     |

**Best model:** Random Forest Classifier

---

## Project Structure

```text
Task4_Disease_Prediction/
│
├── dataset/
│   └── heart.csv
│
├── notebook/
│   └── Disease_Prediction.ipynb
│
├── model/
│   └── heart_disease_model.pkl
│
├── images/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── accuracy_comparison.png
│
├── app.py              # Streamlit web application
├── api.py              # FastAPI REST inference service
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

---

## Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Modeling:** Scikit-Learn (Random Forest, Logistic Regression, SVM)
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## Run Locally

1. Clone the repository:

```bash
git clone https://github.com/shubhambana/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Run the FastAPI backend (optional):

```bash
uvicorn api:app --reload
```

The API docs will be available at `http://127.0.0.1:8000/docs`.

---

## Limitations

- Only accuracy is reported. For medical screening, recall (missed positive cases) and precision should also be evaluated.
- The dataset is small and public, so performance on other patient populations is untested.
- The model is for learning and demonstration purposes only, not for clinical use.

---

## Future Improvements

- Add XGBoost and LightGBM classifiers.
- Train and validate on larger medical datasets.
- Tune the classification threshold based on clinical cost-benefit trade-offs.

---

## Author

**Shubham Singh Rajput**
Aspiring AI & Machine Learning Engineer