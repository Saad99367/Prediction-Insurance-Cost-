import streamlit as st
import joblib
from sklearn.preprocessing import PolynomialFeatures
from PIL import Image



img = Image.open("dataset-cover.jpg")
st.image(img, caption="Welcome", use_container_width=True)

model = joblib.load('insurance_model.pkl')

st.title('Medical Insurance Cost')

age = st.number_input("Age", value=None, placeholder='Enter Your Age')
bmi = st.number_input("Enter Your BMI", value=None, placeholder="Enter BMI")
children = st.number_input("Enter Number of Children", value=None, placeholder="Children")

sex = st.selectbox('Select Gender', ['Male', 'Female'])
region = st.selectbox('Select Your Region', ['northeast', 'northwest', 'southeast', 'southwest'])
smoker = st.radio('Are you Smoker ?', ['Yes', 'No'])

gender_map = {"Male": 1, "Female": 0}
sex_num = gender_map[sex]

smoker_map = {"Yes": 1, "No": 0}
smoker_num = smoker_map[smoker]

region_map = {
    'northeast': 0,
    'northwest': 1,
    'southeast': 2,
    'southwest': 3
}
region_num = region_map[region]

if st.button('Predict'):

    if None in [age, bmi, children]:
        st.warning("Please fill all numeric fields")

    else:
        x_input = [[age, sex_num, bmi, children, smoker_num, region_num]]

        poly = PolynomialFeatures(degree=2)
        poly.fit(x_input)

        x_input_poly = poly.transform(x_input)

        pred = model.predict(x_input_poly)

        st.success(f'Prediction Cost is: {pred[0]:.2f}')
