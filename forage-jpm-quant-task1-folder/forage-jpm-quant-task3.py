### JPMorgan Quatitative Research Forage Task 3

# the aim of this code is to build a predictive model that estimates the probability of default
# of customers based on their characteristics. This code will specifically run a deicision tree
# classification model to help predict the possible losses.

# import libraries
import numpy as np
import pandas as pd
import seaborn as sns

# load the data given by risk manager
df = pd.read_csv('Task 3 and 4_Loan_data.csv')
# print(df.head())

# no categorical variables that must be encoded

# define features and target
features = ['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 
            'income', 'years_employed', 'fico_score', 'default']
X = df[features]
y = df['default']

# split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train tree classification model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

# print evaluation metrics to check legitamacy of tree
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
# print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Import graphviz and export the decision tree to dot format for visualization
import graphviz
from sklearn.tree import export_graphviz

# Generate and display the decision tree graph
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(15,10))
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=["No Default", "Default"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Loan Default Prediction")
plt.show()
