# 🏠 House Price Prediction & Market Analytics

An internship-ready **Data Analytics + Machine Learning + Business Intelligence** project built with Python and Streamlit.

The application transforms housing data into **KPIs, market insights, model explanations, predictions, risks, opportunities and recommended actions**.

## 🎯 Business Problem

Housing organizations need a practical way to understand what factors are associated with property values and to produce a quick, data-driven estimate for a housing profile.

This project answers:

> **Which housing characteristics are most important, what does the market data tell us, and what house value does the model estimate for a selected profile?**

## 📊 Dashboard Modules

1. **Executive Dashboard** — KPIs, management summary, value distribution and top drivers.
2. **Market Insights** — correlations, geographic context and value segments.
3. **Model Performance** — MAE, RMSE, R², actual-vs-predicted chart and feature importance.
4. **Interactive Price Prediction** — enter a housing profile and receive a model estimate.

## 🔍 Analytics Workflow

**Data → Exploration → KPIs → Drivers → ML Model → Evaluation → Prediction → Business Action**

## 🧠 Machine Learning

A `RandomForestRegressor` is trained using an 80/20 train-test split with a fixed random seed for reproducibility.

Evaluation metrics:

- **MAE** — average absolute prediction error
- **RMSE** — emphasizes larger prediction errors
- **R²** — proportion of test-set variation explained by the model

## 🗂 Dataset

The project uses the **California Housing dataset** provided through scikit-learn. It contains housing and demographic variables including median income, house age, average rooms, average bedrooms, population, average occupancy, latitude and longitude, with median house value as the target.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

The application loads the dataset automatically, so no CSV file is required in the repository.

## 🛠 Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## 📁 Repository Structure

```text
ibmBobProject/
├── house_price_app.py
├── requirements.txt
├── PROJECT_REPORT.md
└── README.md
```

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run house_price_app.py
```

The application will open in your browser.

## 💼 Business Insights & Actions

### Opportunity
Use the strongest model drivers to prioritize deeper market research and property screening.

### Risk
The model is trained on historical data and may be less reliable for unusual properties or changing market conditions.

### Recommended action
Use predictions as an initial decision-support signal and manually review important or unusual cases.

## ⚠️ Disclaimer

This project is an educational/internship analytics application. Model predictions are estimates and **must not be treated as professional real-estate valuations or financial advice**.

## 👨‍💻 Internship Project

**Repository:** `LRBishnoi/ibmBobProject`

Built as a practical demonstration of data analytics, machine learning and business intelligence.
