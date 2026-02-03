import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Title and Description
st.title('🐧 Palmer Penguin Species Predictor')
st.markdown("""
This app predicts the **Penguin Species** (Adelie, Chinstrap, or Gentoo) 
using a machine learning model trained on the **Palmer Penguins dataset**.
""")
st.image("https://allisonhorst.github.io/palmerpenguins/reference/figures/lter_penguins.png") 

# 2. Load and Clean Data
@st.cache_data
def load_data():
    # Load the specific CSV from your Kaggle download
    df = pd.read_csv('penguins_size.csv')
    
    # Drop rows with missing values (NaN) to prevent errors
    df.dropna(inplace=True)
    
    # Optional: Fix a known issue in this dataset where one 'sex' entry is just a "."
    df = df[df['sex'] != '.']
    
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'penguins_size.csv' not found. Please make sure the file is in the same folder.")
    st.stop()

# 3. Train Model (on the fly)
# We separate the "Target" (Species) from the "Features" (Measurements)
X = df.drop(columns=['species'])
y = df['species']

# Convert categorical text (Male/Female, Island) into numbers
X = pd.get_dummies(X, columns=['island', 'sex'])

# Train the Random Forest
model = RandomForestClassifier()
model.fit(X, y)

# 4. Sidebar Inputs for User
st.sidebar.header('User Input Features')

def user_input_features():
    # Slider inputs for numerical features
    island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    sex = st.sidebar.selectbox('Sex', ('MALE', 'FEMALE'))
    bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)
    
    # Store inputs in a DataFrame
    data = {'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 5. Preprocess User Input (Match the training data format)
# We join user input with the original data (dummy row) to ensure One-Hot Encoding matches
# (This trick ensures we get columns like 'island_Dream' even if the user picked 'Biscoe')
raw_penguins = df.drop(columns=['species'])
combined_df = pd.concat([input_df, raw_penguins], axis=0)

# Encode just like we did for training
encode = ['island', 'sex']
combined_df = pd.get_dummies(combined_df, columns=encode)

# Select only the first row (the user input)
input_row = combined_df[:1]

# 6. Prediction
st.subheader('Prediction')
prediction = model.predict(input_row)
prediction_proba = model.predict_proba(input_row)

st.write(f"The model predicts this penguin is a: **{prediction[0]}**")
st.write("---")

# 7. Visualization (Feature Importance)
# This shows the user WHICH inputs mattered most
st.subheader('Why did the model choose this?')
st.write("Feature Importance (What the AI looked at):")

feature_importance = model.feature_importances_
feature_names = input_row.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
st.pyplot(fig)