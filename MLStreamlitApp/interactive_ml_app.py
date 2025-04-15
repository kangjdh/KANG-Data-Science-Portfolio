import streamlit as st
import pandas as pd
import numpy as np
#from sklearn.datasets import fetch_california_housing
#import seaborn as sns

# temporary title CHANGE IT LATER
# add emojis
st.title('Interactive Machine Learning App')


# subheader / brief intro to app's purpose
st.subheader(":star2: Welcome to my interactive machine learning app!:star2:") 
st.write("Where you can take a deeper dive into understanding " \
"your dataset through :orange[***supervised machine learning***] " \
"with your model of choice.")
st.write("Let's get straight to it!")

st.divider()

## uploading dataset
st.subheader(":eject: **Upload Your Data**", divider=True)

# initialize df
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# allow users to add dataset of choice
custom_dataset = st.file_uploader(':violet-badge[Step 1:] Upload the .csv dataset file here:', type=['csv'])
if custom_dataset is not None:
    st.session_state.dataset = pd.read_csv(custom_dataset)

# add dataset options
st.write('Here are some example datasets to try!')

col1, col2, col3, = st.columns(3)
with col1:
    if st.button('California Housing:house:', key='house'):
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
        dataset['target'] = data.target
        st.session_state.dataset = dataset
with col2:
    if st.button('Titanic:boat:', key='boat'):
        import seaborn as sns
        dataset = sns.load_dataset('titanic')
        st.session_state.dataset = dataset
with col3:
    if st.button('Iris:eye:'):
        from sklearn.datasets import load_iris
        data = load_iris()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
        dataset['target'] = data.target
        st.session_state.dataset = dataset

# button to preview chosen dataset
if st.session_state.dataset is not None:
    if st.button('Dataset Preview', type='primary'):
        st.dataframe(st.session_state.dataset.head())

## supervised ML model
st.subheader(":point_up_2: **Choose a Supervised Machine Learning Model**", divider=True)
st.write("Once you've uploaded your dataset, choose a supervised machine learning model" \
" to use to ***djsfkdjslkfjskldfjs***:")

# adjust features & choose target feature
st.write('but first, lets make the data useable')
st.write(':blue-badge[Step 2:] Adjust features as needed:')


## performance feedback
st.subheader(":green-badge[Step 3:] **Review Performance Feedback**", divider=True)






