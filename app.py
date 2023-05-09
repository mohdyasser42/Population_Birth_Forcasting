import streamlit as st
!pip install streamlit_lottie
import json
from streamlit_lottie import st_lottie
from population_birth_forcasting import predict 

st.set_page_config(page_title="Population and Birth Prediction",page_icon= "🤖",layout="wide")

animation = st.container()
heading = st.container()

with heading:
    st.markdown("<h1 style='text-align: center; margin: -2rem 0 1rem 0;'>Population and Birth Forcasting</h1>", unsafe_allow_html=True)

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
    elif int(year_input) < 2022:
        st.write("Please enter a valid year from 2023.")
    elif int(year_input) > 2100:
        st.write("Please enter a valid year less than 2100.")
    else:
        year = int(year_input)
        prediction = predict(country,year)
        st.write(f"The prediction from 2023 to {year} is:") 
        st.write(prediction)
        st.write("---")
        st.markdown("<h2 style='text-align: center; margin-bottom: 1rem;'>Visualization</h2>", unsafe_allow_html=True)
        with st.container():
            left_column, right_column = st.columns(2)
            with left_column:
                st.write("Total Population Graph (in Thousands):")
                st.bar_chart(prediction['Total Population'])
                st.write("Gender Population Graph (in Thousands):")
                st.bar_chart(prediction[['Male Population','Female Population',]])
                st.write("Gender Ratio in Population Graph:")
                st.bar_chart(prediction['Population Gender Ratio'])
                st.write("Birth Rate Graph:")
                st.bar_chart(prediction['Birth Rate'])


            with right_column:       
                st.write("Total Births Graph (in Thousands):")
                st.bar_chart(prediction['Total Births'])
                st.write("Gender Births Graph (in Thousands):")
                st.bar_chart(prediction[['Male Births','Female Births']])
                st.write("Gender Ratio in Births Graph:")
                st.bar_chart(prediction['Birth Gender Ratio'])
                

        
