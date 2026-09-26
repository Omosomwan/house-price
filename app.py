import streamlit as st
from house_price import get_model, get_town, get_state_town
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import joblib

# Title
st.set_page_config(page_title='House Price', layout='centered',)
col1, col2 = st.columns(2)
with col1:
    st.image('images.jpg', width=250)
    
with col2:
    st.title('House Price in Nigeria')


st.divider()
    
 

# Input
state = st.selectbox('State', ['Lagos', 'Abuja', 'Ogun', 'Oyo', 'Rivers', 'Imo', 'Enugu', 'Edo',
                    'Delta', 'Akwa Ibom', 'Kaduna'])
town_list = get_town(state)
town = st.selectbox('Town', town_list)

bedrooms = st.selectbox('Bedrooms', [1,2,3,4,5,6,7,8,9,10])
bathrooms = st.selectbox('Bathrooms', [1,2,3,4,5,6,7,8,9,10])
parking_space = st.selectbox('Parking Space', [1,2,3,4,5,6,7,8,9,10])
title_list = ['Detached Duplex', 'Terraced Duplexes', 'Semi Detached Duplex','Detached Bungalow',
               'Block of Flats', 'Semi Detached Bungalow','Terraced Bungalow']
title = st.selectbox('Parking Space', title_list)

state_df = get_state_town(state, town)

X = state_df.drop(columns = ['price', 'toilets'])
y = state_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

input_data = pd.DataFrame(
    {
        'bedrooms': [bedrooms],
        'bathrooms' : [bathrooms],
        'parking_space' : [parking_space],
        'title' : [title]
    },
    
)
st.divider()

st.table(input_data)

# Training the model
processor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output= False, drop='first'),
         ['title']),
        ('num', StandardScaler(), ['bedrooms','bathrooms', 'parking_space'])
    ],
  remainder='passthrough'  
)

pipe = Pipeline(
    [
        ('processor', processor),
        ('model', GradientBoostingRegressor(n_estimators=50))
    ]
)
model = pipe.fit(X_train, y_train)

# Output     
if st.button('Predict'):
    with st.spinner('Thinking...'):
        time.sleep(2)
        prediction = model.predict(input_data)
        st.balloons()
        st.success(f'Housing price: ₦{prediction[0]:,.0f}')
st.divider()

# Building AI assistant
import langchain
from langchain_openai import ChatOpenAI
from APIKEY import key
st.title('AI Assistant')

prompt = st.text_input("Ask me any Question on Housing")
llm = ChatOpenAI(
        model="openai/gpt-4o-mini",  # add openai/ prefix for OpenRouter

    base_url="https://openrouter.ai/api/v1", # this tells it to use OpenRouter

    api_key=key
)


if prompt:
    # Use LangChain's tuple format: (role, content)
    with st.spinner('Thinking..'):
        messages = [
            ("system", 'You are an real estate AI assistant. You only provide information on housing'),
            ("human", prompt)
        ]
        
        # Pass the formatted messages list to the model
        response = llm.invoke(messages)
        st.write(response.content)

