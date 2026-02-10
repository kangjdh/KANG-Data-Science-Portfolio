### JPMorgan Quatitative Research Forage Task 1

# this code aims to take the csv file "nat_gas.csv," which lists the price of gas at the end of each
# month from Oct 2020 to Sep 2024, and create a supervised machine learning code that takes dates for 
# inputs and returns past prices or predicted price for up to a year.

# check for linearity to see which regression line would be appropriate
# this will be commented so the graph is not produced when users run the script
#import seaborn as sns
#import matplotlib.pyplot as plt

#plt.figure(figsize=(12,6))
#sns.scatterplot(x=df['Dates'], y=df['Prices'])
#plt.title("Gas Prices Over Time")
#plt.xlabel("Date")
#plt.ylabel("Gas Prices")
#plt.xticks(rotation=45)
#plt.show()

# since pronounced curvature can be seen, polynomial regression will be used to fit the data

# import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

# import dataframe from same folder
df = pd.read_csv('nat_gas.csv')

# 'Dates' must be converted into a format readable to Python as numerical input
# use datetime from pandas to do this
df['Dates'] = pd.to_datetime(df['Dates'])
# convert dates to numerical values 
df['Dates_Ordinal'] = df['Dates'].map(datetime.toordinal)

# no other data tidying is required as there is no missing data nor other non-numerical variables
# data also does not have to be scaled since there is only one variable, 'Dates'

# define X and y variables / target
X = pd.DataFrame(df['Dates_Ordinal'])
y = pd.Series(df['Prices'])

# center the feature to keep X values smaller
X_centered = X - X.mean()

# split the raw data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_centered, y, test_size = 0.2, random_state = 42)

# create new feature matrix will all polynomial combinations up to degree 3
# the degree 3 was derived from running a loop that finds the degree from 1 to 20 that 
# leads to the smallest mean squared error, root mean squared error, and r^2 score
poly = PolynomialFeatures(degree=3, include_bias=False)
#poly_features = poly.fit_transform(X.values.reshape(-1,1))
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# initialize and train the linear regression model on unscaled 
# polynomial regression is a type of linear regression
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# make predictions on the test set
y_pred = poly_reg.predict(X_test_poly)

# predict gas prices for the next year using polynomial regression line created above
# add on another year to x-axis and convert to ordinal as with data above
last_date = df['Dates'].max()
next_year_date = last_date.replace(year=last_date.year + 1)
next_year_ordinal = next_year_date.toordinal()

# center data with mean since ordinal numbers are large
X_mean = X.mean().values[0]
next_year_centered = np.array([[next_year_ordinal - X_mean]])

# transform to polynomial features
next_year_poly = poly.transform(next_year_centered)

# predict next year price
next_year_price = poly_reg.predict(next_year_poly)

print(f"Automatically predicted gas price on {next_year_date.strftime('%m/%d/%Y')}: ${next_year_price[0]:.2f}")

min_date = df['Dates'].min()

# allow user to input date of choice
# the estimated price of the gas will be given in return
# Expect date is in MM/DD/YYYY format
def predict_price(user_date):
    try:
        # convert text string to date
        input_date = datetime.strptime(user_input.strip(), '%m/%d/%Y')
        # set error note if date is less than range
        if input_date < min_date:
            return f"Date inputed is too early. Please enter a date on or after {min_date.strftime('%m/%d/%Y')}."
        # set error note if date is more than range
        elif input_date > next_year_date:
            return f"Date inputed is too far in the future. Please enter a date on or before {next_year_date.strftime('%m/%d/%Y')}."
        # convert date to ordinal value and center as with other data already in regression graph
        input_ordinal = input_date.toordinal()
        input_centered = np.array([[input_ordinal - X_mean]])
        # apply polynomial features on inputed date and predict based on data
        input_poly = poly.transform(input_centered)
        predicted_price = poly_reg.predict(input_poly)
        return f"Predicted gas price on {input_date.strftime('%m/%d/%Y')}: ${predicted_price[0]:.2f}"
    # if error, raise ValueError asking for correct date format
    except ValueError:
        return "Invalid date format. Please use MM/DD/YYYY."

# allow users to input their own date
user_input = input(f"Enter a date between {min_date.strftime('%m/%d/%Y')} and {next_year_date.strftime('%m/%d/%Y')}: ")
print(predict_price(user_input))
