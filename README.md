# 🏠 House Price Prediction & Market Analytics

An internship-ready **Data Analytics + Machine Learning + Business Intelligence** project built with Python.

## Project Description

The project analyzes the California Housing dataset to identify price patterns and important drivers, evaluate a Random Forest regression model, and provide a supporting house-value estimate.

## Dashboard / Analysis Modules

1. **Executive Dashboard** — KPIs, management summary, value distribution, top drivers and business interpretation.
2. **Market Insights** — correlations, geographic context, income-band value trend and market segments.
3. **Model Performance** — MAE, RMSE, R², actual-vs-predicted analysis and feature importance.
4. **Price Prediction** — model-based estimated median house value for a selected housing profile.

## Analytics Workflow

**Data → Exploration → KPIs → Trends → Drivers → ML Model → Evaluation → Prediction → Business Action**

## Machine Learning

A `RandomForestRegressor` is trained using an 80/20 train-test split with `random_state=42`.

Evaluation metrics:

- **MAE** — average absolute prediction error
- **RMSE** — emphasizes larger prediction errors
- **R²** — proportion of test-set variation explained by the model

## Dataset

The project uses the **California Housing dataset** provided through scikit-learn.

Dataset documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

The dataset is loaded automatically; no separate CSV is required.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Submission Files

```text
ibmBobProject/
├── LaduRam_HousePricePrediction.ipynb
├── requirements.txt
├── LaduRam_ProjectReport.docx
└── README.md
```

These are the four files named according to the internship submission instructions.

## Setup / Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the code notebook:

```bash
jupyter notebook
```

Open `LaduRam_HousePricePrediction.ipynb` and run the code cell from top to bottom.

## Business Insights

### Opportunity
Use the strongest model drivers for market screening and deeper analysis.

### Risk
Historical data may not represent current market conditions, and unusual properties may produce less reliable estimates.

### Recommended Action
Use model predictions as decision-support estimates and manually review important or unusual cases.

## Disclaimer

This is an educational/internship project. Model predictions are estimates based on historical data and are not professional real-estate valuations or financial advice.

## Repository

**GitHub:** `LRBishnoi/ibmBobProject`
