import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#loading and preparing the data 
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#training the and testing the model


model = RandomForestClassifier() 
model.fit(X_train, y_train)       
# Make predictions on the test set
y_pred = model.predict(X_test)

# Check how well the model performs
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy on test data:", accuracy)

# Streamlit code starts here
st.title("Iris Flower Type Prediction")
st.write("This app predicts the type of Iris flower based on measurements you enter.")

# Input sliders for each measurement
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Predict button and output
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write("The model predicts this is an Iris of type:", iris.target_names[prediction[0]])
