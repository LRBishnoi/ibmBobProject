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
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df["MedHouseVal"] = df["MedHouseVal"] * 100000
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    }
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, metrics, importance

df = load_data()
model, metrics, importance = train_model(df)

st.title("🏠 House Price Prediction & Market Analytics")
st.caption("Internship project: turning housing data into insights, predictions, risks and actions.")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Executive Overview", "Market Analysis", "Prediction"])

if page == "Executive Overview":
    st.subheader("Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties", f"{len(df):,}")
    c2.metric("Average House Value", f"${df.MedHouseVal.mean():,.0f}")
    c3.metric("Median House Value", f"${df.MedHouseVal.median():,.0f}")
    c4.metric("Model R²", f"{metrics['R2']:.2%}")

    st.markdown("### Key business insights")
    top_feature = importance.index[0]
    high_value_share = (df["MedHouseVal"] >= df["MedHouseVal"].quantile(0.75)).mean()
    insights = [
        f"**Primary driver:** `{top_feature}` has the highest feature importance in the Random Forest model.",
        f"**Premium segment:** approximately {high_value_share:.1%} of observations are in the top 25% of house values.",
        f"**Model quality:** the regression model explains about {metrics['R2']:.1%} of test-set price variation.",
        "**Recommended action:** use the model as a screening and valuation-support tool, then review unusual cases manually.",
    ]
    for item in insights:
        st.markdown(f"- {item}")

    st.markdown("### Price distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df["MedHouseVal"], bins=40, kde=True, ax=ax)
    ax.set_xlabel("Median House Value ($)")
    ax.set_ylabel("Number of Areas")
    st.pyplot(fig)
    plt.close(fig)

elif page == "Market Analysis":
    st.subheader("Sales & Market Analysis")
    st.write("Explore the variables associated with housing values.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Feature importance")
        fig, ax = plt.subplots(figsize=(7, 5))
        importance.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Importance")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.markdown("#### Correlation with house value")
        corr = df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal").sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        corr.plot(kind="barh", ax=ax)
        ax.set_xlabel("Correlation")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### Data sample")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Risk, opportunity and action")
    st.info("**Risk:** model estimates may be less reliable for unusual properties or market conditions not represented in the historical data.")
    st.success("**Opportunity:** the strongest model drivers can help analysts prioritize which housing characteristics deserve deeper market investigation.")
    st.warning("**Action:** use predicted values as a decision-support signal, not as a final professional valuation.")

else:
    st.subheader("Interactive House Price Prediction")
    st.write("Enter a housing profile to generate an estimated median house value.")

    cols = st.columns(4)
    values = {}
    defaults = {
        "MedInc": 3.5,
        "HouseAge": 28.0,
        "AveRooms": 5.5,
        "AveBedrms": 1.1,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 35.0,
        "Longitude": -119.0,
    }
    for i, feature in enumerate(defaults):
        with cols[i % 4]:
            low = float(df[feature].min())
            high = float(df[feature].max())
            values[feature] = st.number_input(feature, min_value=low, max_value=high, value=float(np.clip(defaults[feature], low, high)))

    input_df = pd.DataFrame([values])
    prediction = model.predict(input_df)[0]
    st.metric("Estimated Median House Value", f"${prediction:,.0f}")
    st.caption("This estimate is produced by the trained Random Forest model and should not be treated as a professional property valuation.")

    st.markdown("### Model performance")
    a, b, c = st.columns(3)
    a.metric("MAE", f"${metrics['MAE']:,.0f}")
    b.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    c.metric("R²", f"{metrics['R2']:.2%}")

st.sidebar.markdown("---")
st.sidebar.caption("Built for internship submission • Data analytics + ML + business intelligence")
