import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error


# Importing the dataset
df = pd.read_csv('nigeria_houses_data.csv')

# Droping the duplicates
df.drop_duplicates(inplace=True)

# Dropping Anambara State
df = df[df['state']!='Anambara']

def get_model(state = "Lagos", town = 'Lekki'):
# Sellecting the state
    state_df = df[df['state'] == state].drop('state', axis = 1)
    

# Sellecting the town
    town = state_df[state_df['town'] == town].drop('town', axis = 1)
    
# Dealing with outliers
    q1 = town['price'].quantile(0.25)
    q3 = town['price'].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 3.0 * IQR
    upper_bound = q3 + 3.0 * IQR
    filt = (town['price'] <=lower_bound) & (town['price'] >=upper_bound)
    town = town[~filt]
    
    X = town.drop(columns = ['price', 'toilets'])
    y = town['price']
    
    # Splitting to X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    processor = ColumnTransformer(
    transformers= [('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['title']),
                   ('scaler', StandardScaler(), ['bedrooms','bathrooms', 'parking_space'])],
    remainder='drop')
    
    # Assemblying the model
    pipe = Pipeline([
    ('processor', processor),
    ('model', GradientBoostingRegressor(n_estimators=100))])
    
    # Train the model
    model = pipe.fit(X_train, y_train)
    
    # Testing the model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    #return f'rmse: {rmse:,.2f}'
    return model
    

def get_town(state = 'Lagos'):
    town = df[df['state'] == state]['town'].str.strip().unique()
    return list(town)
get_town()

def get_state_town(state = 'Lagos', town = 'Lekki'):
    state_df = df[df['state'] == state].drop('state', axis = 1)
    town_df = state_df[state_df['town'] == town]. drop('town', axis = 1)
    
    q1 = town_df['price'].quantile(0.25)
    q3 = town_df['price'].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q1 - 0.25 * IQR
    upper_bound = q3 + 0.25 * IQR
    filt = (town_df['price'] <=lower_bound) | (town_df['price'] >=upper_bound)
    town = town_df[~filt]
    
    return town