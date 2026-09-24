# House Price Prediction & Market Analytics

An internship-ready data analytics and machine learning project that turns housing data into business-friendly insights and a simple house-price prediction tool.

## Project objective

Use housing characteristics to understand market patterns, identify the main price drivers, visualize regional trends, and predict the median house value for a selected property profile.

## What this project demonstrates

- Data loading and preparation
- Exploratory data analysis
- KPI-based business dashboard
- Trend and driver analysis
- Correlation analysis
- Machine learning regression
- Model evaluation
- Interactive house-price prediction
- Actionable recommendations

## Dataset

The project uses the California Housing dataset distributed through scikit-learn. It contains census-derived California housing information and a target representing median house value.

Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

The dataset is loaded automatically by the application, so no separate CSV download is required.

## Files

- `house_price_app.py` - single-file Streamlit application containing the analytics dashboard and ML model.
- `requirements.txt` - Python dependencies.
- `PROJECT_REPORT.md` - project report/concept note that can be converted to DOCX or PDF for final submission.
- `README.md` - project overview and setup instructions.

## How to run

```bash
pip install -r requirements.txt
streamlit run house_price_app.py
```

The application opens in a browser and provides an overview dashboard, market analysis, model performance, and an interactive prediction section.

## Business Intelligence flow

**Data -> Information -> Insights -> Decision -> Action**

Examples of possible actions include prioritizing high-value locations, identifying properties with unusually high or low predicted values, and using the model as a supporting tool for initial property valuation.

## Important note

Predictions are estimates from a historical dataset and should not be treated as professional real-estate valuations.
