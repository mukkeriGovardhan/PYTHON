import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("Linear Regression Model")

# Load dataset
data = pd.read_csv("data.csv")

st.subheader("Dataset")
st.write(data)

# Features and Target
X = data[['hours']]
y = data['score']

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Model Coefficients")
st.write("Intercept:", model.intercept_)
st.write("Slope:", model.coef_[0])

# Prediction input
hours = st.number_input("Enter Study Hours", min_value=0.0, max_value=24.0)

if st.button("Predict Score"):
    prediction = model.predict([[hours]])
    st.success(f"Predicted Score: {prediction[0]:.2f}")

# Plot regression
fig, ax = plt.subplots()

ax.scatter(X, y)
ax.plot(X, model.predict(X))

ax.set_xlabel("Hours Studied")
ax.set_ylabel("Score")

st.pyplot(fig)