import streamlit as st
import json
from streamlit_lottie import st_lottie
from forecasting_model import predict 

st.set_page_config(page_title="Population and Birth Forecasting",page_icon= "📈",layout="wide")

animation = st.container()
heading = st.container()

with heading:
    st.markdown("<h1 style='text-align: center; margin: -2rem 0 1rem 0;'>Population and Birth Forecasting</h1>", unsafe_allow_html=True)

def load_lottiefile(path:str):
    with open(path,"r") as p:
        return json.load(p)

pop = load_lottiefile("./population.json")

with animation:
    st_lottie(pop, height= 250)

with st.form("prediction_form"):
    country = st.selectbox("Select country:",("India","Asia","World","South America", "Europe", "Africa" ,"Oceania" , "Australia", "United States of America", "Brazil","United Kingdom") )
    year_input = st.text_input("Enter a year (YYYY):", max_chars=4, key="year_input")
    submit_button = st.form_submit_button(label="Predict")

st.write("---")

if submit_button and year_input:
    if not year_input.isnumeric():
        st.write("Please enter a valid integer year.")
    elif int(year_input) < 2024:
        st.write("Please enter a valid year from 2024.")
    elif int(year_input) > 2100:
        st.write("Please enter a valid year less than 2100.")
    else:
        year = int(year_input)
        prediction = predict(country,year)
        st.write(f"The prediction from 2023 to {year} is:") 
        st.write(prediction)
        st.write("---")
        col = prediction.columns
        def avg_increase(population_data):
            differences = []
            for i in range(1, len(population_data)):
                diff = population_data[i] - population_data[i-1]
                differences.append(diff)
            avg_increase = sum(differences) / (len(population_data) - 1)
            avg_increase_percentage = (avg_increase / population_data[0]) * 100
            return avg_increase_percentage
        value = {}
        for i in range(1,len(col)):
            bp = prediction[col[i]].values
            result = round(avg_increase(bp),4)
            value[f"value_{i}"] = result
        start_yr = prediction.index[0]
        end_yr = prediction.index[-1]
        st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>Visualization</h2>", unsafe_allow_html=True)
        with st.container():
            left_column, right_column = st.columns(2)
            with left_column:
                st.write("Total Population Graph (in Thousands):")
                st.bar_chart(prediction['Total Population'])
                st.write(f"➣ The Total Population in {country} is expected to experience an average annual increase of {value['value_1']}% from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")
                st.write("Gender Population Graph (in Thousands):")
                st.bar_chart(prediction[['Male Population','Female Population']])
                st.write(f"➣ The Gender Population in {country} is expected to experience an average annual increase of {value['value_2']}% in Male and {value['value_3']}% in Female from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")
                st.write("Gender Ratio in Population Graph:")
                st.bar_chart(prediction['Population Gender Ratio'])
                st.write(f"➣ The Gender Ratio of Population in {country} is expected to experience an average annual increase of {value['value_4']}% from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")
                st.write("Birth Rate Graph:")
                st.bar_chart(prediction['Birth Rate'])
                st.write(f"➣ The Birth Rate in {country} is expected to experience an average annual change of {value['value_9']}% from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")



            with right_column:       
                st.write("Total Births Graph (in Thousands):")
                st.bar_chart(prediction['Total Births'])
                st.write(f"➣ The Total Births in {country} is expected to experience an average annual increase of {value['value_5']}% from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")
                st.write("Gender Births Graph (in Thousands):")
                st.bar_chart(prediction[['Male Births','Female Births']])
                st.write(f"➣ The Gender Births in {country} is expected to experience an average annual increase of {value['value_6']}% in Male and {value['value_7']}% in Female from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")
                st.write("Gender Ratio in Births Graph:")
                st.bar_chart(prediction['Birth Gender Ratio'])
                st.write(f"➣ The Gender Ratio of Births in {country} is expected to experience an average annual increase of {value['value_8']}% from {start_yr} to {end_yr} according to forecasts.")
                st.write("---")
                

        
