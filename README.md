# Bank Customer Churn Predictor

An end-to-end machine learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN), with a live Streamlit web app for real-time predictions.

---

## Problem Statement

Banks lose significant revenue when customers close their accounts and move to competitors. This project builds a binary classifier that identifies customers at risk of churning — giving the bank an opportunity to intervene with targeted retention strategies before it's too late.

**Dataset:** [Churn Modelling — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) — 10,000 bank customers across France, Germany, and Spain.

---

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | **86.80%** |
| Class-1 Recall (churners found) | **0.49** |
| Class-1 Precision | **0.78** |
| AUC baseline (always-predict-stay) | ~79.6% |

**Confusion Matrix (2,000 test customers):**

|  | Predicted: Stay | Predicted: Leave |
|---|---|---|
| **Actual: Stay** | 1,538 ✅ | 55 ❌ |
| **Actual: Leave** | 209 ❌ | 198 ✅ |

> Class-1 recall is intentionally low (~49%) due to class imbalance (80% stay / 20% leave). See the Bonus section in the notebook for class-weight experiments that improve it.

---

## Model Architecture

```
Input (13 features)
    │
    ▼
Dense(16, ReLU)      ← learns simple patterns
    │
    ▼
Dense(8, ReLU)       ← combines into larger patterns
    │
    ▼
Dense(1, Sigmoid)    ← outputs churn probability (0–1)
```

**Training config:** Adam optimizer · Binary Cross-Entropy loss · 50 epochs · Batch size 32 · EarlyStopping (patience=5)

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Neural Network | TensorFlow / Keras |
| Preprocessing | scikit-learn (ColumnTransformer, OneHotEncoder, StandardScaler) |
| Data | pandas, NumPy |
| Visualization | Matplotlib |
| Model Persistence | joblib |
| Web App | Streamlit |
| Language | Python 3.10+ |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/armankhan8150/bank-customer-churn-ann.git
cd bank-customer-churn-ann
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

The app loads the pre-trained `model.keras` and `preprocessor.pkl` automatically. No retraining needed.

### 4. (Optional) Retrain from scratch

Open `train.ipynb` in Jupyter and run all cells. The notebook saves fresh `model.keras` and `preprocessor.pkl` files.

---

## App Preview

![Bank Customer Churn Predictor App](assets\app_screenshot.png)

The app collects 10 customer features via input widgets and predicts churn probability in real time.

**Test these two customers manually:**

| Feature | Customer A (→ STAY) | Customer B (→ LEAVE) |
|---|---|---|
| Age | 38 | 60 |
| Geography | France | Germany |
| Gender | Male | Female |
| Active Member | Yes | No |
| Balance | Average | 120,000 |

---

## Training Loss Curve

![Training vs Validation Loss](assets\loss_plot.png)

Both curves decrease together without diverging — the model is generalising, not overfitting.

---

## Pipeline Summary

1. **Load** — read CSV, check shape (10,000 × 14)
2. **Clean** — drop `RowNumber`, `CustomerId`, `Surname`; confirm 0 missing values
3. **Split** — 80% train / 20% test, stratified by churn label
4. **Preprocess** — OneHotEncode Geography + Gender; StandardScale all numeric features (fit on train only)
5. **Build** — `Sequential` ANN: Input(13) → Dense(16, ReLU) → Dense(8, ReLU) → Dense(1, Sigmoid)
6. **Train** — 50 epochs, batch 32, EarlyStopping on `val_loss`
7. **Evaluate** — accuracy, confusion matrix, classification report
8. **Save** — `model.keras` + `preprocessor.pkl` via joblib
9. **Deploy** — Streamlit app reloads both artefacts and predicts on new inputs

---

## License

MIT License — see [LICENSE](LICENSE) for details.
