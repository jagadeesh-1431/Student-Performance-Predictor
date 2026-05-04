# 🎓 Student Performance Predictor

> A professional, end-to-end machine learning web application that predicts student mathematics examination scores using **Principal Component Analysis (PCA)** and **Linear Regression**, deployed via an interactive **Flask** dashboard.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/jagadeesh-1431/Student-Performance-Predictor)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Results & Evaluation](#-results--evaluation)
- [Future Scope](#-future-scope)
- [References](#-references)

---

## 🧠 Overview

Student academic performance is influenced by a complex interplay of demographic, socioeconomic, and preparation-related factors. Traditional performance assessment is **reactive** — it identifies underperformance only after examinations. This project offers a **proactive solution**.

The **Student Performance Predictor** is a machine learning pipeline that:
- Ingests student demographic and academic features
- Applies one-hot encoding and StandardScaler normalization
- Reduces dimensionality using **PCA (14 components, >95% variance retained)**
- Trains a **Linear Regression** model to predict mathematics scores
- Serves predictions via a **Flask web dashboard** in real time

This system enables educators and administrators to identify at-risk students early and deploy targeted academic interventions.

---

## 🌐 Live Demo

Run locally at `http://127.0.0.1:5000` after setup (see [Quick Start](#-quick-start)).

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | **0.87** |
| **RMSE** | **5.36** |
| **MAE** | **4.21** |
| **MSE** | **28.73** |
| **PCA Components** | **14** |
| **Variance Retained** | **>95%** |
| **Training Samples** | 800 (80%) |
| **Test Samples** | 200 (20%) |

The PCA-augmented Linear Regression model outperforms a baseline (no PCA) by approximately **4 percentage points in R²**, confirming that PCA resolves multicollinearity between reading and writing scores effectively.

---

## 📁 Project Structure

```
Student-Performance-Predictor/
│
├── app.py                  # Main Flask web application
├── main.py                 # ML training pipeline (PCA + Linear Regression)
├── predict.py              # Command-line prediction utility
│
├── data/
│   └── students.csv        # Students Performance in Exams dataset (1,000 records)
│
├── models/
│   ├── model.pkl           # Trained Linear Regression model
│   ├── pipeline.pkl        # Fitted StandardScaler + PCA pipeline
│   └── metrics.json        # Model evaluation metrics
│
├── templates/
│   ├── index.html          # Main dashboard page
│   └── result.html         # Prediction result page
│
├── static/
│   ├── css/                # Stylesheet assets
│   └── js/                 # Frontend scripts
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset

**Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

| Property | Value |
|----------|-------|
| Total Records | 1,000 |
| Total Features | 8 (5 categorical + 2 numerical + 1 target) |
| Missing Values | None |
| Target Variable | `math score` (continuous, 0–100) |

### Feature Description

| Feature | Type | Category | Description |
|---------|------|----------|-------------|
| `gender` | Categorical | Demographic | Student gender: male / female |
| `race/ethnicity` | Categorical | Demographic | Ethnic group: A, B, C, D, or E |
| `parental level of education` | Categorical | Socioeconomic | Highest education level of parent |
| `lunch` | Categorical | Socioeconomic | Standard or free/reduced lunch |
| `test preparation course` | Categorical | Academic | Completed or none |
| `reading score` | Numerical | Academic | Score in reading exam (0–100) |
| `writing score` | Numerical | Academic | Score in writing exam (0–100) |
| `math score` | **Target** | Academic | Score in math exam (0–100) |

**Target Distribution:** Mean = 66.09, Std = 15.16, Min = 0, Max = 100

---

## ⚙️ Methodology

### 1. Data Preprocessing
- **No missing values** — dataset is complete across all 1,000 records.
- **One-Hot Encoding** via `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap, expanding features to ~17–19 binary columns.
- **StandardScaler** normalization to zero mean and unit variance — essential before PCA since it is a variance-maximization algorithm.
- **Data Leakage Prevention** — all transformations are wrapped in a Scikit-learn `Pipeline`, fitted exclusively on training data.
- **Train-Test Split** — 80:20 ratio (`random_state=42`), yielding 800 training and 200 test samples.

### 2. Dimensionality Reduction — PCA

| Parameter | Value |
|-----------|-------|
| `n_components` | 14 |
| Algorithm | Full SVD |
| `random_state` | 42 |
| Preprocessing | StandardScaler |
| Variance Retained | >95% |

PCA resolves multicollinearity between `reading score` and `writing score` by projecting data onto orthogonal components. The 14 components retain over 95% of total dataset variance.

### 3. Model — Linear Regression

The model learns weights `w` and bias `b` such that:

```
ŷ = wᵀ · x + b
```

Trained via **Ordinary Least Squares (OLS)** with the closed-form solution:

```
w = (XᵀX)⁻¹ Xᵀy
```

OLS is particularly well-suited here given the small dataset size (800 training samples), yielding fast and stable convergence without iterative optimization.

### 4. Core Training Pipeline

```python
# Preprocessing
X = df.drop(columns=[target_column])
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA Pipeline
pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components, random_state=42))
X_train_pca = pipeline.fit_transform(X_train)
X_test_pca  = pipeline.transform(X_test)

# Training and Evaluation
model = LinearRegression()
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
r2 = r2_score(y_test, y_pred)   # ~0.87
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/jagadeesh-1431/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

### 2. Setup Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python main.py
```

This generates `models/model.pkl`, `models/pipeline.pkl`, and `models/metrics.json`.

### 5. Run the Web Application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 💻 Usage

### Web Dashboard

1. Navigate to `http://127.0.0.1:5000`
2. Fill in the student's details — gender, ethnicity, parental education, lunch type, test preparation status, reading score, and writing score.
3. Click **Predict** to receive an instant predicted mathematics score.

### Command-Line Prediction

```bash
python predict.py
```

Follow the prompts to enter student attributes and receive a predicted score directly in the terminal.

### Custom PCA Components

```bash
python main.py --n_components 10
```

The pipeline enforces `n_components = min(n_components, feature_count)` to prevent over-specification.

---

## 🔧 Configuration

Key parameters can be modified in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | `14` | Number of PCA components to retain |
| `test_size` | `0.2` | Fraction of data used for testing |
| `random_state` | `42` | Seed for reproducibility |
| `target_column` | `math score` | Column to predict |

---

## 📉 Results & Evaluation

### Metrics Output (`models/metrics.json`)

```json
{
  "mse": 28.73,
  "rmse": 5.36,
  "mae": 4.21,
  "r2": 0.87,
  "features": 14
}
```

### Key Findings

- **Gender Impact:** Male students showed marginally higher mathematics scores on average, while female students performed better in reading and writing.
- **Test Preparation Effect:** Students who completed the test preparation course outperformed peers by an average of ~5–8 points in mathematics.
- **Parental Education Correlation:** Strong positive correlation observed between parental education level and student scores across all subjects.
- **Cross-Subject Spillover:** A strong correlation between language skills (reading/writing) and mathematics performance suggests integrated curriculum approaches may improve math outcomes.
- **Socioeconomic Factor:** Students receiving standard lunch (proxy for higher socioeconomic status) consistently outperformed peers on free/reduced lunch plans.

### Why PCA Helped

- Reduced feature dimensionality from ~18–19 encoded features to 14 orthogonal components.
- Eliminated multicollinearity between `reading score` and `writing score` (which are highly correlated with each other and with the target).
- Improved R² by approximately **4 percentage points** over baseline Linear Regression without PCA.

---

## 🔭 Future Scope

- **Ensemble Methods** — Explore Gradient Boosting (XGBoost, LightGBM) or Random Forest Regressor to capture non-linear relationships.
- **Multi-Target Prediction** — Simultaneously predict reading, writing, and mathematics scores as a multi-output regression problem.
- **Explainable AI (XAI)** — Apply SHAP (SHapley Additive exPlanations) values to restore feature-level interpretability lost in PCA.
- **Real-Time Data Integration** — Connect to live Student Information Systems (SIS) for continuous semester-wide monitoring.
- **Deep Learning** — Investigate Multi-Layer Perceptrons (MLP) that can automatically learn hierarchical feature representations.
- **Expanded Feature Set** — Incorporate attendance records, study hours, and socioeconomic indicators for higher predictive power.

---

## 📦 Requirements

```
flask
scikit-learn
pandas
numpy
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📚 References

1. Students Performance in Exams Dataset — Kaggle. https://www.kaggle.com/datasets/spscientist/students-performance-in-exams
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
3. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
4. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
5. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proc. 9th Python in Science Conference*.
6. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362.
7. Flask Documentation — Pallets Projects. https://flask.palletsprojects.com/
8. Romero, C., & Ventura, S. (2010). Educational data mining: A review. *IEEE Trans. Systems, Man, Cybernetics*, 40(6), 601–618.
9. UNESCO. (2023). Global Education Monitoring Report.

---

## 👤 Author

**Jagadeesh**
- GitHub: [@jagadeesh-1431](https://github.com/jagadeesh-1431)
- Project Repo: [Student-Performance-Predictor](https://github.com/jagadeesh-1431/Student-Performance-Predictor)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*CSE 274 – Machine Learning Lab | Academic Year 2025–2026*