import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import variation

# Load and prepare data
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
bias = lr.score(X_train, y_train)
variance = lr.score(X_test, y_test)
cv = variation(df['target'])
y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean)**2)
intercept = lr.intercept_

# Streamlit UI
st.title("🧠 Diabetes Progression Predictor using Linear Regression")

st.subheader("📊 Dataset Preview")
st.write(df.head())

st.subheader("📈 Model Performance Metrics")
st.markdown(f"- **R² Score (Test Set):** {r2:.3f}")
st.markdown(f"- **Mean Squared Error (MSE):** {mse:.3f}")
st.markdown(f"- **Mean Absolute Error (MAE):** {mae:.3f}")
st.markdown(f"- **Bias (Training R²):** {bias:.3f}")
st.markdown(f"- **Variance (Testing R²):** {variance:.3f}")
st.markdown(f"- **Coefficient of Variation (Target):** {cv:.3f}")
st.markdown(f"- **Sum of Squares for Regression (SSR):** {SSR:.2f}")
st.markdown(f"- **Model Intercept:** {intercept:.2f}")

# Feature Importance
st.subheader("🔍 Feature Coefficients")
coeff_df = pd.DataFrame(lr.coef_, index=X.columns, columns=["Coefficient"]).sort_values(by="Coefficient", ascending=False)
st.dataframe(coeff_df)

# Plot Actual vs Predicted
st.subheader("📉 Actual vs Predicted Plot")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)

# High Risk Patients
st.subheader("⚠️ High-Risk Patients (Predicted Target > 150)")
high_risk = X_test.copy()
high_risk['Actual'] = y_test
high_risk['Predicted'] = y_pred
high_risk = high_risk[high_risk['Predicted'] > 150]
st.dataframe(high_risk)

# Footer
st.markdown("---")
st.caption("🔧 Built with ❤️ using Streamlit and scikit-learn.")
