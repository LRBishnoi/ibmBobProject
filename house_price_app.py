import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="House Price Analytics", page_icon="🏠", layout="wide")

@st.cache_data
def load_data():
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df["MedHouseVal"] = df["MedHouseVal"] * 100000
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions),
    }
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions})
    return model, metrics, importance, results

df = load_data()
model, metrics, importance, results = train_model(df)
q1, q2, q3 = df["MedHouseVal"].quantile([0.25, 0.50, 0.75])

st.title("🏠 House Price Prediction & Market Analytics")
st.caption("Internship project: Data → Insights → Prediction → Action")

with st.sidebar:
    st.header("Project Navigation")
    page = st.radio("Select module", ["Executive Dashboard", "Market Insights", "Model Performance", "Price Prediction"])
    st.divider()
    st.markdown("**Dataset:** California Housing")
    st.markdown("**Model:** Random Forest Regressor")
    st.markdown("**Purpose:** Decision support")

if page == "Executive Dashboard":
    st.subheader("Executive Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Housing Areas", f"{len(df):,}")
    c2.metric("Average Value", f"${df.MedHouseVal.mean():,.0f}")
    c3.metric("Median Value", f"${q2:,.0f}")
    c4.metric("Model R²", f"{metrics['R2']:.1%}")

    st.markdown("### Management summary")
    top_feature = importance.index[0]
    st.markdown(
        f"The analysis covers **{len(df):,} housing observations**. The median observed house value is **${q2:,.0f}**. "
        f"The strongest model driver is **{top_feature}**, while the Random Forest model achieves an R² of **{metrics['R2']:.1%}** on the test set."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### House-value distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["MedHouseVal"], bins=40, kde=True, ax=ax)
        ax.set_xlabel("Median House Value ($)")
        ax.set_ylabel("Number of Areas")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Top model drivers")
        fig, ax = plt.subplots(figsize=(7, 4))
        importance.head(8).sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Feature importance")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Business interpretation")
    st.success("**Opportunity:** use the strongest price drivers to prioritize deeper market research and property screening.")
    st.warning("**Risk:** historical data may not reflect current market conditions, so predictions require human review.")
    st.info("**Action:** use the model as an initial valuation-support signal, not as a replacement for professional appraisal.")

elif page == "Market Insights":
    st.subheader("Market Insights")
    st.write("Explore relationships, segments and value patterns in the housing data.")

    left, right = st.columns(2)
    corr = df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal").sort_values()
    with left:
        st.markdown("#### Correlation with house value")
        fig, ax = plt.subplots(figsize=(7, 5))
        corr.plot(kind="barh", ax=ax)
        ax.set_xlabel("Correlation")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Geographic context")
        map_df = df[["Latitude", "Longitude"]].rename(columns={"Latitude": "lat", "Longitude": "lon"})
        st.map(map_df)
        st.caption("Geographic distribution of the observations.")

    st.markdown("### Income-band value trend")
    trend = df.copy()
    trend["Income Band"] = pd.qcut(trend["MedInc"], 10, duplicates="drop")
    trend_summary = trend.groupby("Income Band", observed=True)["MedHouseVal"].mean().reset_index()
    trend_summary["Income Midpoint"] = trend_summary["Income Band"].apply(lambda x: (x.left + x.right) / 2)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend_summary["Income Midpoint"], trend_summary["MedHouseVal"], marker="o")
    ax.set_xlabel("Median income (dataset units)")
    ax.set_ylabel("Average median house value ($)")
    ax.set_title("Average house value across income bands")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Market segments")
    df_view = df.copy()
    df_view["Value Segment"] = pd.qcut(df_view["MedHouseVal"], 4, labels=["Value", "Mid-Low", "Mid-High", "Premium"])
    segment = df_view.groupby("Value Segment", observed=True)["MedHouseVal"].agg(["count", "mean"]).reset_index()
    segment.columns = ["Segment", "Areas", "Average Value"]
    st.dataframe(segment.style.format({"Average Value": "${:,.0f}"}), use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.subheader("Model Performance")
    a, b, c = st.columns(3)
    a.metric("MAE", f"${metrics['MAE']:,.0f}")
    b.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    c.metric("R²", f"{metrics['R2']:.2%}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Actual vs predicted")
        sample = results.sample(min(1500, len(results)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sample["Actual"], sample["Predicted"], alpha=0.35)
        limits = [sample.min().min(), sample.max().max()]
        ax.plot(limits, limits, linestyle="--")
        ax.set_xlabel("Actual value ($)")
        ax.set_ylabel("Predicted value ($)")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Feature importance")
        fig, ax = plt.subplots(figsize=(7, 5))
        importance.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Interpretation")
    st.write("MAE represents average absolute prediction error, RMSE gives greater weight to larger errors, and R² indicates the proportion of test-set variation explained by the model.")

else:
    st.subheader("Interactive Price Prediction")
    st.write("Enter a housing profile to estimate its median house value.")

    defaults = {"MedInc": 3.5, "HouseAge": 28.0, "AveRooms": 5.5, "AveBedrms": 1.1,
                "Population": 1200.0, "AveOccup": 3.0, "Latitude": 35.0, "Longitude": -119.0}
    values = {}
    input_cols = st.columns(4)
    for i, feature in enumerate(defaults):
        low, high = float(df[feature].min()), float(df[feature].max())
        with input_cols[i % 4]:
            values[feature] = st.number_input(feature, min_value=low, max_value=high,
                                              value=float(np.clip(defaults[feature], low, high)))

    prediction = model.predict(pd.DataFrame([values]))[0]
    st.markdown("### Estimated value")
    st.metric("Predicted Median House Value", f"${prediction:,.0f}")

    if prediction >= q3:
        st.success("This estimate falls in the premium-value range of the dataset.")
    elif prediction <= q1:
        st.info("This estimate falls in the value-oriented range of the dataset.")
    else:
        st.info("This estimate falls within the middle range of observed values.")

    st.caption("Important: this is a model estimate based on historical data, not a professional real-estate valuation.")

st.divider()
st.caption("Internship project • Python • Pandas • Scikit-learn • Streamlit • Business Intelligence")
