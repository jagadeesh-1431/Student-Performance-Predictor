# Student Performance Predictor

A professional, easy-to-use web application for predicting student academic performance using Machine Learning (PCA + Linear Regression).

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Model**:
   ```bash
   python main.py
   ```

4. **Run Web App**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

## 📁 Project Structure

- `app.py` — Main Flask web application
- `main.py` — Machine learning training pipeline
- `predict.py` — Simple command-line prediction utility
- `data/` — Contains `students.csv` dataset
- `models/` — Saved ML models and metrics
- `templates/` — HTML dashboard files
- `static/` — CSS and UI assets
- `requirements.txt` — Project dependencies

## 📊 Methodology

The system uses:
- **PCA (Principal Component Analysis)** for dimensionality reduction.
- **Linear Regression** for score prediction.
- **Flask** for the interactive dashboard.
