import pickle

import streamlit as st
import pandas as pd
import os

import numpy as np
import altair as alt

model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

st.title('Prediksi Harga Mobil')
st.image('car.jpg')

#open file csv
df1 = pd.read_csv("car.csv")

view = st.sidebar.selectbox('Tampilkan data', ['Dataset', 'Grafik Highway-mpg', 'Grafik curbweight', 'Grafik horsepower'])

if view == 'Dataset':
    st.header("Dataset")
    st.dataframe(df1)

elif view == 'Grafik Highway-mpg':
    st.write("Grafik Highway-mpg")
    chart_highwaympg = df1['highwaympg']
    st.line_chart(chart_highwaympg)

elif view == 'Grafik curbweight':
    st.write("Grafik curbweight")
    chart_curbweight = df1['curbweight']
    st.line_chart(chart_curbweight)

elif view == 'Grafik horsepower':
    st.write("Grafik horsepower")
    chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
    st.line_chart(chart_horsepower)

#input nilai dari variable independent
highwaympg = st.number_input('highwaympg')
curbweight = st.number_input('curbweight')
horsepower = st.number_input('horsepower')

if st.button('Prediksi'):
    #prediksi variable yang telah diinputkan
    car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

    # convert ke string
    harga_mobil_str = np.array(car_prediction)
    harga_mobil_float = float(harga_mobil_str)

    #tampilkan hasil prediksi
    st.write(f'Prediksi harganya adalah {harga_mobil_float:.2f}')