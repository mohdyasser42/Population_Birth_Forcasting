# **Importing All The Neccessary Modules**
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scikit-learn.model_selection import train_test_split
from scikit-learn.preprocessing import StandardScaler
from scikit-learn.linear_model import LinearRegression
from scikit-learn.model_selection import GridSearchCV

def predict(country_name, year):
    # DATA PREPARATION
    # Converting the Dataset file into a pandas dataframe
    df = pd.read_csv('WPP2022_Demographic_Indicators_Medium.csv')
    # FEATURE SELECTION
    Main_df = df[['Location', 'Time','TPopulation1July','TPopulationMale1July','TPopulationFemale1July','PopSexRatio','Births','SRB','CBR']]
    # SAMPLE SELECTION
    # Selecting years from 1950 to 2022
    Main_df = Main_df[(Main_df['Time'] >= 1950) & (Main_df['Time'] <= 2022)]
    # Selecting only limited countries for now
    selected_countries = ["Asia", "World", "South America", "Europe", "Africa" ,"Oceania" , "Australia", "India", "United States of America", "Brazil","United Kingdom" ]
    new_df = Main_df[Main_df['Location'].isin(selected_countries)].copy()
    new_df.reset_index(drop=True, inplace=True)
    new_df['Location'] = new_df['Location'].str.lower()

    # FEATURE EXTRACTION
    # Deriving values of births in gender
    def birthgenders(srb, births):
        denom = (srb/100)+1
        births_f = births/denom
        births_m = births-births_f
        return births_m,births_f

    country_name = country_name.lower()

    # Creating new columns for total male and female births and storing the values after deriving
    new_df[['BirthM', 'BirthF']] = new_df.apply(lambda row: pd.Series(birthgenders(row['SRB'], row['Births'])), axis=1)
    cols = list(new_df.columns)
    cols.insert(7, cols.pop(cols.index('BirthM')))
    cols.insert(8, cols.pop(cols.index('BirthF')))
    new_df = new_df.reindex(columns=cols)

    # Creating a Dataframe for the Input of a Country
    country_wanted = new_df[new_df['Location'] == country_name]
    country_wanted

    # Selecting the Dependent and Indepnedent Attributes for the Model
    x_cols_drop = ['Location','TPopulation1July','TPopulationMale1July','TPopulationFemale1July','PopSexRatio','Births','BirthM','BirthF','SRB','CBR']
    y_cols_drop = ['Location','Time','TPopulation1July','PopSexRatio','Births','SRB','CBR']

    x = country_wanted.drop(columns=x_cols_drop)
    y = country_wanted.drop(columns=y_cols_drop)

    # Creating Training and Testing Attributes for the Model
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

    # FEATURE SCALING 
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(x_train_scaled, columns = x_train.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns = x_test.columns)

    # Using Linear Regression Model and Applying Hyperparameter Tuning Using GridSearchCV
    fit_intercept = [True,False]
    copy_X = [True,False]
    n_jobs = [-1,1]
    positive = [True,False]

    param_grid = {
        'fit_intercept': fit_intercept,
        'copy_X': copy_X,
        'n_jobs': n_jobs,
        'positive': positive
    }

    lr=LinearRegression()
    grid_search=GridSearchCV(estimator=lr,param_grid=param_grid,cv=10,n_jobs=-1)
    grid_search.fit(x_train_scaled,y_train)

    best_grid=grid_search.best_estimator_

    model =best_grid.fit(x_train_scaled,y_train)

    # Converting the input year into Dataframe of years from current year to input year

    year_range = range(2023,year+1)

    input_years = np.array(year_range)

    x_pred = pd.DataFrame({'Time': input_years})

    scaler2 = StandardScaler()
    scaler2.fit(x_test)
    x_pred_scaled = scaler2.transform(x_pred)

    # Pridicting The Dependent Values
    predict_test = model.predict(x_pred_scaled)

    # Extracting the Other Attributes Using Predicted Dependent Values of the given Country name and Year
    pred_MPop = np.round(predict_test[:,0],3)
    pred_FPop = np.round(predict_test[:,1],3)
    pred_TPop = np.round(predict_test[:,0] + predict_test[:,1],3)
    pred_PopSR = np.round((predict_test[:,0] / predict_test[:,1])*100,3)
    pred_MBirth = np.round(predict_test[:,2],3)
    pred_FBirth = np.round(predict_test[:,3],3)
    pred_Tbirths = np.round(predict_test[:,2] + predict_test[:,3],3)
    pred_SRB = np.round((predict_test[:,2] / predict_test[:,3])*100,3)
    pred_CBR = np.round((pred_Tbirths / pred_TPop)*1000,3)

    # Predicted Values are Stored in the DataFrame
    predicted_values = {
        'Location' : country_name,
        'Time' : input_years,
        'Total Population' : pred_TPop,
        'Male Population' : pred_MPop,
        'Female Population' : pred_FPop,
        'Population Gender Ratio' : pred_PopSR,
        'Total Births' : pred_Tbirths,
        'Male Births' : pred_MBirth,
        'Female Births' : pred_FBirth,
        'Birth Gender Ratio' : pred_SRB,
        'Birth Rate' : pred_CBR
    }
    predicted_df = pd.DataFrame(predicted_values) # this dataframe contains the all predicted values
    predicted_df['Time'] = predicted_df['Time'].astype(str)
    predicted_df = predicted_df.set_index('Time')
    return(predicted_df)
