import streamlit as st
import pandas as pd

st.title("A Deep Dive with Palmer's Penguins")

st.subheader("This app allows for a deeper look into a population of penguins.")
st.write('Explore the features of the population based on various criteria!')

df = pd.read_csv("data\penguins.csv")

# multi-selectbox for specifying species
species = st.sidebar.multiselect("Select a species:", 
                                 options=df['species'].unique(), 
                                 default=df['species'].unique())

# multi-selectbox for specifying island(s)
island = st.sidebar.multiselect('Select an island:', 
                                options=df['island'].unique(),
                                default=df['island'].unique())

# slider determining minimum and maxiumum bill length in mm
bill_length_mm = st.sidebar.slider('Select a range of bill lengths (mm):', 
                                   30, 60, (30, 60))

# slider determining minimum and maxiumum bill depth in mm
bill_depth_mm = st.sidebar.slider('Select a range of bill depths (mm):', 
                                   10, 25, (10, 25))

# slider determining minimum and maximum flipper length
flipper_length_mm = st.sidebar.slider('Select a range for flipper lengths (mm):', 
                                   170, 235, (170, 235))

# slider determining minimum and maximum body mass (grams)
body_mass_g = st.sidebar.slider('Select a range for body masses (grams):', 
                                   2700, 6300, (2700, 6300))

# multi-select box for selecting penguin sex
sex = st.sidebar.multiselect('Select a sex:', 
                             options=df['sex'].unique(),
                             default=df['sex'].unique())

# input years to filter out based on penguin years
st.sidebar.write('Select years:')
opt_1 = st.sidebar.checkbox('2007')
opt_2 = st.sidebar.checkbox('2008')
opt_3 = st.sidebar.checkbox('2009')
year = []
if opt_1:
    year.append(2007)
if opt_2:
    year.append(2008)
if opt_3:
    year.append(2009)


# Filter the dataframe based on selections
filtered_df = df[df['species'].isin(species) & 
                 df['island'].isin(island) &
                 df['bill_length_mm'].between(bill_length_mm[0], bill_length_mm[1]) &
                 df['bill_depth_mm'].between(bill_depth_mm[0], bill_depth_mm[1]) &
                 df['flipper_length_mm'].between(flipper_length_mm[0], flipper_length_mm[1]) &
                 df['body_mass_g'].between(body_mass_g[0], body_mass_g[1]) & 
                 df['sex'].isin(sex) &
                 df['year'].isin(year)]

# Display the filtered dataframe
if filtered_df.empty:
    st.write('No results matched the selected criteria... Try a new combo!')
else:
    st.dataframe(filtered_df)