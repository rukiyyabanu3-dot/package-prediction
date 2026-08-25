import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/CS-06/OneDrive/place.csv")

x = data.iloc[:, [-2]].values
y = data.iloc[:, [-1]].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

ls = LinearRegression()
ls.fit(x_train, y_train)

st.title("Package Prediction")

cgpa = st.number_input(
    "Enter CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.01
)

if st.button("Predict Package"):
    package = ls.predict([[cgpa]])

    st.write("CGPA:", cgpa)
    st.write("Predicted Package:", round(package[0][0], 2), "LPA")
