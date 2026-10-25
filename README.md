# loan-approval-prediction-ml
Machine Learning-based Loan Approval Prediction using Python, Scikit-learn, Random Forest, and Streamlit. Includes EDA, feature engineering, model comparison, hyperparameter tuning, evaluation metrics, and an interactive prediction interface.
# 🏦 Loan Approval Prediction Using Machine Learning

A machine learning classification project that predicts loan outcomes based on applicant demographics, financial information, employment experience, and credit history. This project explores multiple classification algorithms and uses a tuned Random Forest Classifier to power an interactive Streamlit application.

## 📌 Project Overview

The objective of this project is to develop a binary classification model that predicts loan status (`0` or `1`) using historical applicant data.

The project covers the complete machine learning workflow, including exploratory data analysis, data preprocessing, feature engineering, model comparison, hyperparameter tuning, model evaluation, feature importance analysis, and an interactive prediction interface.

**Note:** This is an educational machine learning project, not a real-world loan approval system or financial advice tool.

## 🎯 Objectives

- Perform exploratory data analysis (EDA) to understand the dataset.
- Investigate class imbalance, duplicate records, outliers, and anomalous values.
- Preprocess numerical and categorical features.
- Train and compare multiple classification algorithms.
- Tune the Random Forest model using cross-validation.
- Evaluate performance using classification metrics and a confusion matrix.
- Analyze feature importance and potential data leakage.
- Develop an interactive prediction interface using Streamlit.

## 📂 Dataset Description

The dataset contains **45,000 records and 14 columns**, including 13 input features and one target variable.

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_gender` | Applicant's gender |
| `person_education` | Education level |
| `person_income` | Applicant's income |
| `person_emp_exp` | Employment experience |
| `person_home_ownership` | Home ownership status |
| `loan_amnt` | Requested loan amount |
| `loan_intent` | Purpose of the loan |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount relative to income |
| `cb_person_cred_hist_length` | Credit history length |
| `credit_score` | Applicant's credit score |
| `previous_loan_defaults_on_file` | Historical loan default information |
| `loan_status` | Binary classification target |

### Target Distribution

The target variable is imbalanced:

| Class | Records | Percentage |
|---|---:|---:|
| 0 | 35,000 | 77.78% |
| 1 | 10,000 | 22.22% |
| **Total** | **45,000** | **100%** |

The exact business meaning of each class should be verified against the original dataset documentation before interpreting predictions as approval or rejection.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas and NumPy** — Data manipulation
- **Matplotlib and Seaborn** — Data visualization
- **Scikit-learn** — Preprocessing, model training, and evaluation
- **XGBoost** — Gradient-boosting classification
- **Joblib** — Model serialization
- **Streamlit** — Interactive web application
- **Jupyter Notebook** — Model development and experimentation

## 🔍 Project Workflow

### 1. Exploratory Data Analysis

- Examined dataset structure, data types, and descriptive statistics.
- Analyzed the target distribution and class imbalance.
- Checked missing values and duplicate records.
- Investigated unusual age and income observations.
- Explored numerical and categorical features.

### 2. Data Preprocessing and Feature Engineering

- Encoded categorical variables.
- Prepared numerical and categorical features for model training.
- Split the dataset into training and testing sets.
- Used an encoded feature matrix containing 27 features.
- Investigated anomalous values and feature relationships.

Preprocessing should be fitted on training data only and reused consistently for validation, testing, and inference.

### 3. Model Comparison

Five classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. XGBoost Classifier

The initial comparison used accuracy, precision, recall, F1-score, and ROC-AUC to understand different aspects of model performance.

### 4. Random Forest Hyperparameter Tuning

RandomizedSearchCV was used to explore Random Forest hyperparameters with five-fold cross-validation.

The selected configuration was:

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 25 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 2 |
| `max_features` | `sqrt` |
| `bootstrap` | `True` |
| Cross-validation folds | 5 |
| Best cross-validation F1-score | 0.829 |

The search optimized F1-score, and the final candidate was evaluated on a held-out test set.

## 📊 Model Performance

The tuned Random Forest produced the following results on the 9,000-record test set.

| Metric | Score |
|---|---:|
| Accuracy | 92.84% |
| ROC-AUC | 97.48% |
| Class 1 Precision | 85.80% |
| Class 1 Recall | 81.30% |
| Class 1 F1-score | 83.00% |
| Cross-validation F1-score | 82.90% |

These are experimental results from the current workflow. They do not guarantee equivalent performance on unseen populations or different loan amounts.

### Confusion Matrix

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 6,730 | 270 |
| Class 1 | 374 | 1,626 |

The confusion matrix helps identify false positives and false negatives, which may have different consequences in a lending context.

## 📈 Feature Importance

Feature importance analysis identified the following influential predictors:

- Previous loan default history
- Loan amount relative to income
- Loan interest rate
- Applicant income
- Home ownership
- Credit score

Permutation importance was also used to examine how shuffling individual encoded features affected the model's F1-score.

These results describe predictive associations learned by the model; they do not establish causation. Importance can also be distributed across correlated features or multiple encoded columns.

## 💻 Interactive Streamlit Application

The project includes a Streamlit interface where users can enter applicant and loan details and receive a prediction from the saved classification model.

### Application Features

- Modern dark-themed user interface
- Applicant and loan information forms
- Automatic loan-to-income ratio calculation
- Random Forest prediction
- Class probability display
- Prediction result visualization

The application is intended for local experimentation and portfolio demonstration. It has not been validated as a production lending service.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or another Python version compatible with the installed libraries
- Git
- A local development environment such as VS Code or Jupyter Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/loan-approval-prediction-ml.git
cd loan-approval-prediction-ml
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

If the saved model bundle is available locally:

```bash
streamlit run app.py
```

The application will open in your browser, usually at `http://localhost:8501`.

**Important:** The application requires a compatible saved model bundle containing the fitted classifier, preprocessing object, feature columns, and categorical options. If the model file is not included in the repository, generate it by running the relevant training and serialization cells in the notebook.

## ⚠️ Limitations and Responsible Use

This project has several limitations that should be considered before interpreting its predictions.

- **Limited training distribution:** The dataset contains loan amounts up to approximately 35,000 in its original units. The model has not been validated for substantially larger amounts, such as 500,000 or 1,000,000.
- **Potential target leakage:** The availability and timing of the loan interest rate must be investigated to determine whether it would be known at prediction time.
- **Class imbalance:** The target distribution is approximately 78% class 0 and 22% class 1, so accuracy alone is insufficient for evaluating performance.
- **Generalization:** Test-set results do not guarantee performance on different applicant populations or future lending conditions.
- **Fairness:** Additional assessment is required to identify potential disparities across relevant applicant groups.
- **Probability calibration:** Predicted class probabilities have not necessarily been calibrated and should not be interpreted as guaranteed chances of approval.
- **Decision impact:** The model is not intended to make actual lending decisions.

This project is for educational and portfolio purposes only. It should not be used to determine eligibility for real financial products.

## 🔮 Future Improvements

- Validate the model on independent and representative data.
- Investigate potential target leakage.
- Improve preprocessing using end-to-end scikit-learn pipelines.
- Compare models using consistent cross-validation.
- Evaluate calibration and decision thresholds.
- Add permutation importance and SHAP explanations.
- Assess fairness and model performance across applicant groups.
- Improve error handling and input validation in the Streamlit application.
- Explore deployment after further validation.

## 👨‍💻 Author

**Ashish Pratap Dwivedi**

Interested in Data Science, Machine Learning, Artificial Intelligence, and data-driven problem-solving.

Connect with me on [LinkedIn](https://www.linkedin.com/) and explore my other projects on [GitHub](https://github.com/).

## 📄 License

Add a license only after deciding how you want others to use, modify, and distribute this project. Also ensure that the dataset's terms permit any intended redistribution.

---

*Built as a practical machine learning project to explore classification, model evaluation, and interactive ML applications.*
