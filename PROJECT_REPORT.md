# House Price Prediction & Market Analytics

## 1. Executive Summary

This project develops a simple analytics and machine-learning application for understanding housing-market patterns and estimating median house values. The application combines exploratory analysis, business KPIs, feature importance, model evaluation and an interactive prediction interface.

## 2. Problem Statement

Housing data contains multiple factors that influence property values. The objective is to transform these variables into understandable insights and build a model that can provide a supporting estimate of median house value for a given housing profile.

## 3. Objectives

1. Prepare and inspect housing data.
2. Identify important variables associated with house value.
3. Present useful KPIs and market insights.
4. Train a regression model for price estimation.
5. Evaluate the model using MAE, RMSE and R².
6. Provide an interactive interface for users.
7. Translate findings into risks, opportunities and recommended actions.

## 4. Dataset

The application uses the California Housing dataset provided by scikit-learn. It contains housing and demographic variables such as median income, house age, average rooms, average bedrooms, population, average occupancy, latitude and longitude, with median house value as the target.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

## 5. Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## 6. Methodology

The workflow is:

**Data → Cleaning/Inspection → Exploratory Analysis → KPI & Driver Analysis → Model Training → Evaluation → Prediction → Business Action**

A Random Forest Regressor is trained using an 80/20 train-test split with a fixed random seed for reproducibility.

## 7. Key KPIs

- Number of observations
- Average house value
- Median house value
- Model R²
- Mean Absolute Error
- Root Mean Squared Error

## 8. Business Insights

Feature importance identifies which variables contribute most strongly to the model's predictions. Correlation analysis provides an additional view of the relationship between individual variables and house value.

The project should be interpreted as a decision-support system. Historical model performance does not guarantee future market accuracy.

## 9. Risk Analysis

- Historical data may not represent current market conditions.
- Unusual properties can produce less reliable estimates.
- Correlation does not prove causation.
- The model should not replace a professional property valuation.

## 10. Opportunities

- Use the model for preliminary valuation screening.
- Identify high-value market segments.
- Investigate the strongest price drivers in greater detail.
- Extend the application with newer local datasets and geographic visualization.

## 11. Recommended Actions

1. Use model estimates as an initial screening signal.
2. Manually review properties with unusual input profiles.
3. Retrain the model when new market data becomes available.
4. Add local and time-based variables before using the system for real operational decisions.

## 12. Conclusion

The project demonstrates how data analytics and machine learning can be combined to convert raw housing data into KPIs, insights, predictions and recommended business actions. Its main value is not simply the prediction model, but the complete decision-oriented workflow.
