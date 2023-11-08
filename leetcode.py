import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
x=iris.data
y=iris.target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


k_near = KNeighborsClassifier(n_neighbors=3)
k_near.fit(x_train, y_train)

# Streamlit app
st.title("Iris Flower Prediction App")

# Sidebar with input fields
st.sidebar.header("Input Parameters")

# Input fields for Sepal Length, Sepal Width, Petal Length, and Petal Width
sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)


user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


prediction = k_near.predict(user_input)


result=prediction[0]
if result==0:
    st.title("Iris-setosa")
elif result==1:
    st.title("Iris-versicolor")
else:
    st.title("Iris-virginica")
