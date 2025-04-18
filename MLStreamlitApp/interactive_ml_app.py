import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# title the app
st.title('Interactive Machine Learning App')


# subheader / brief intro to app's purpose
st.subheader(":star2: Welcome to my interactive machine learning app!:star2:") 
st.write("...where you can take a deeper dive into understanding " \
"your dataset through :orange[***supervised machine learning***] " \
"with your model of choice and evaluate the model itself with ease.")
st.write("Let's get straight to it! :blush:")

st.divider()

## uploading dataset
st.subheader(":eject: **Upload Your Data**", divider=True)

# initialize df
if 'df' not in st.session_state:
    st.session_state.df = None

# allow users to add dataset of choice
custom_dataset = st.file_uploader(':violet-badge[Step 1:] **Upload your .csv dataset file here**:', type=['csv'])
if custom_dataset is not None:
    st.session_state.df = pd.read_csv(custom_dataset)

# add dataset options
st.write('Here are some example datasets to try out!')

col1, col2, col3, = st.columns(3)
with col1:
    if st.button('California Housing:house:', key='house'):
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.session_state.df = df
with col2:
    if st.button('Titanic:boat:', key='boat'):
        df = sns.load_dataset('titanic')
        st.session_state.df = df
with col3:
    if st.button('Iris:eye:'):
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        st.session_state.df = df

st.write(':red[**Note:**] of the example datasets, only the "Titanic" dataset can be applied to all' \
            ' three model options given.')

dataset = st.session_state.df

# button to preview chosen dataset
if dataset is not None:
    if st.button('Dataset Preview', type='primary'):
        st.dataframe(dataset.head())
        st.write(dataset.shape)
        st.caption('The numbers within the parantheses above indicates number of rows' \
        ' and number of columns in the dataset, respectively')

## supervised ML model
st.subheader(":point_up_2: **Choose a Supervised Machine Learning Model**", divider='green')
st.write("Now that you've uploaded your dataset, the next step is to select a supervised" \
" machine learning model to unconver the meaning behind the mountain of numbers.")

# adjust features & choose target
st.write("But first, let's determine what information is going to be used and found.")
st.write(':blue-badge[Step 2:] **Select features and target:**')

if dataset is not None:
   
    features_columns = st.multiselect("Select features (X):", dataset.columns.tolist())
    st.caption('The ***features*** are the columns from the dataset that will be used' \
    ' in the supervised machine learning model.')
    
    target_column = st.selectbox("Select target (y):", dataset.columns.tolist())
    st.caption('The ***target*** is the column that the machine learning model' \
    ' will aim to find/calculate.')

st.write(':blue-badge[Step 2.1:] **Adjust features (if necessary):**')

# drop rows to handle missing values
st.write("You may want to drop rows with missing values based on their quantity per" \
" feature. Here is a list of features with their number of missing values.")

if dataset is not None:
# check how many missing values each feature has
    missing_values = dataset[features_columns].isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        st.write("There are no missing values in the dataset! :smile:")
    else:
        st.dataframe(missing_values.reset_index().rename(columns={
            'index': 'Feature',
            0: 'Quantity of Missing Values'}))
        
    drop_checkbox = st.multiselect('Choose features to drop missing values for:',
                                   options = missing_values.index)
    if drop_checkbox:
        dataset = dataset.dropna(subset=drop_checkbox)

# define X and y
    if features_columns and target_column:
        X = dataset[features_columns]
        y = dataset[target_column]

# encode categorical variables
        categoricals = X.select_dtypes(include=['object', 'category']).columns
        if len(categoricals) > 0:
            st.warning(f':warning: Categorical column(s) {list(categoricals)} detected and encoded :warning:')
            st.caption('The columns mentioned above were determined to have non-numeric variables and were ' \
            ' appropriately switched to numbers to allow proper calculations for the models.')
        X = pd.get_dummies(X, drop_first=True)

    if st.button('Features Preview', type = 'primary', key='frog'):
        st.dataframe(X.head())
        st.write(X.shape)

st.markdown("---")

st.write(':green-badge[Step 3:] **Choose a model to apply to your dataset:**')

# display the buttons side by side
col_model_1, col_model_2, col_model_3 = st.columns(3)

if 'run_linreg' not in st.session_state:
    st.session_state.run_linreg = False
if 'run_logreg' not in st.session_state:
    st.session_state.run_logreg = False
if 'run_dt' not in st.session_state:
    st.session_state.run_dt = False

with col_model_1:
    if st.button('Linear Regression (Scaled)'):
        st.session_state.run_linreg = True
        st.session_state.run_logreg = False
        st.session_state.run_dt = False
with col_model_2:
    if st.button("Logistic Regression"):
        st.session_state.run_logreg = True
        st.session_state.run_linreg = False
        st.session_state.run_dt = False
with col_model_3:
    if st.button('Decision Tree'):
        st.session_state.run_dt = True
        st.session_state.run_linreg = False
        st.session_state.run_logreg = False

def del_prev_results(exclude=None):
    model_keys = {
        'lin_reg_scaled': ['mse', 'rmse', 'r2', 'model_coef'],
        'logreg_model': ['logreg_model', 'logreg_accuracy', 'logreg_y_test', 'logreg_y_pred', 'logreg_cm', 'logreg_report', 'logreg_coef', 'logreg_intercept'],
        'dt_model': ['dt_model', 'dt_accuracy', 'dt_y_test', 'dt_y_pred', 'dt_X_train', 'dt_cm', 'dt_report']}
    for model, keys in model_keys.items():
        if model != exclude:
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]

from sklearn.model_selection import train_test_split

# add linear regression option
if st.session_state.run_linreg:
# delete results from previous models
    del_prev_results(exclude='lin_reg_scaled')
# download necessary libraries
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
# initialize scaler and apply to features
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns = X.columns)
# split the data (80% train, 20% test)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
        X_scaled, y, test_size = 0.2, random_state= 42)
# fit the scaled data
    lin_reg_scaled = LinearRegression()
    lin_reg_scaled.fit(X_train_scaled, y_train_scaled)
# make the predictions
    y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)
# store metrics and results
    st.session_state.mse = mean_squared_error(y_test_scaled, y_pred_scaled)
    st.session_state.rmse = mean_squared_error(y_test_scaled, y_pred_scaled, squared=False)
    st.session_state.r2 = r2_score(y_test_scaled, y_pred_scaled)
    st.session_state.lin_reg_scaled = lin_reg_scaled
    st.session_state.model_coef = pd.Series(lin_reg_scaled.coef_, index=X.columns)
# show success button if linear regression was carried out
    st.success(":white_check_mark: Linear Regression model trained successfully!:white_check_mark:")

# add logistic regression option
if st.session_state.run_logreg:
# delete results from previous models
    del_prev_results(exclude='logreg_model')
# download necessary libraries
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, roc_auc_score
# split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)
# initialize and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
# predict on test data
    y_pred = model.predict(X_test)
# calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
# get the predicted probabilities for the positive class (survival)
    y_probs = model.predict_proba(X_test)[:, 1]
# calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# compute the Area Under the Curve (AUC) score
    roc_auc = roc_auc_score(y_test, y_probs)
# store metrics and results
    st.session_state.logreg_model = model
    st.session_state.logreg_accuracy = accuracy
    st.session_state.logreg_y_test = y_test
    st.session_state.logreg_y_pred = y_pred
    st.session_state.logreg_cm = confusion_matrix(y_test, y_pred)
    st.session_state.logreg_report = classification_report(y_test, y_pred, output_dict=True)
    st.session_state.logreg_coef = pd.Series(model.coef_[0], index=X.columns)
    st.session_state.logreg_intercept = model.intercept_[0]
    st.session_state.logreg_fpr = fpr
    st.session_state.logreg_tpr = tpr
    st.session_state.logreg_roc_auc = roc_auc
# show success button if logistic regression was carried out
    st.success(':white_check_mark: Logistic Regression model trained successfully!:white_check_mark:')

# add decision tree model option
if st.session_state.run_dt:
# delete results from previous models
    del_prev_results(exclude='dt_model')
# download necessary libraries
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.metrics import roc_curve, roc_auc_score
# add a max depth setting hyperparameters
    max_depth = st.number_input('Enter the desired max depth: ', min_value=1, max_value=20)
    st.caption('This input will be the number of times the data is split, and thus the levels of the decision tree.')
    if max_depth:
        del_prev_results(exclude='dt_model')
# split dataset into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)
# Initialize and train tree classification model
    model = DecisionTreeClassifier(random_state = 42, max_depth=max_depth)
    model.fit(X_train, y_train)
# Predict on test data
    y_pred = model.predict(X_test)
# Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
# get the predicted probabilities for the positive class
    y_probs = model.predict_proba(X_test)[:, 1]
# calculate the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
# compute the Area Under the Curve (AUC) score
    roc_auc = roc_auc_score(y_test, y_probs)    
# store metrics and results
    st.session_state.dt_model = model
    st.session_state.dt_accuracy = accuracy
    st.session_state.dt_y_test = y_test
    st.session_state.dt_y_pred = y_pred
    st.session_state.dt_X_train = X_train
    st.session_state.dt_cm = confusion_matrix(y_test, y_pred)
    st.session_state.dt_report = classification_report(y_test, y_pred, output_dict=True)
    st.session_state.dt_roc_auc = roc_auc
    st.session_state.dt_fpr = fpr
    st.session_state.dt_tpr = tpr
# show success button if logistic regression was carried out
    st.success(':white_check_mark: Decision Tree model trained successfully!:white_check_mark:')

# explain each model choice briefly
st.markdown('''
            - :blue-background[Linear Regression] : the simplest of this app's options, this model uses the linear 
            regression equation to discover how each factor impacts the probability of the outcome to what extent. "Scaled"
            means that the numbers are adjusted so that larger numbers don't necessarily mean larger impact.
            - :green-background[Logistic Regression] : similar to linear regression, logistic regression calculates
            what factors impact the probability of the outcome for a binary category using the logistic regression
            equation
            - :red-background[Decision Tree] : is a model that makes decisions by asking a series of yes/no questions,
            designed to mimic the human decision making process''') 
st.markdown("---")


## performance feedback
st.subheader(":mag: **Review Performance Feedback**", divider='orange')

st.write(':orange-badge[Step 4:] **Check out the model-derived results:**')

# print metrics for linear regression
if 'model_coef' in st.session_state:
# show model coefficients
    st.markdown('##### Model Coefficients (Scaled)')
    coef_df = st.session_state.model_coef.reset_index()
    coef_df.columns = ['Feature', 'Coefficient']
    st.dataframe(coef_df.style.background_gradient(cmap="Blues"))
    st.write(f"**Intercept:** {st.session_state.lin_reg_scaled.intercept_:.3f}")
    st.caption(f'The :blue[***coefficients***] next to each feature name indicates the impact that the feature \
               has on the outcome. The higher the value, the greater the impact.')
    st.caption(f'The :blue[***intercept***] is the value of the outcome given that all the features are set to zero.')

# print metrics for logistic regression
if 'logreg_model' in st.session_state:
# show model coefficients
    st.markdown('##### Model Coefficients')
    coef = pd.Series(st.session_state.logreg_model.coef_[0], index = X.columns)
    intercept = st.session_state.logreg_model.intercept_[0]
    st.dataframe(st.session_state.logreg_coef.rename("Coefficient").to_frame().style.background_gradient(cmap='Blues'))
    st.write("Intercept:", st.session_state.logreg_intercept)
    st.caption(f'The :blue[***coefficients***] next to each feature name indicates the impact that the feature \
               has on the outcome. The higher the value, the greater the impact.')
    st.caption(f'The :blue[***intercept***] is the value of the outcome given that all the features are set to zero.')

# print metrics for decision tree
if 'dt_accuracy' in st.session_state:
# import graphviz and export the decision tree to dot format for visualization
    import graphviz
    from sklearn import tree
    st.markdown('##### Decision Tree Visualization')
    dot_tree = tree.export_graphviz(st.session_state.dt_model, out_file=None,
                    feature_names = st.session_state.dt_X_train.columns,
                    class_names = ["Outcome 1", "Outcome 2"],
                    filled = True, rounded = True)
    st.graphviz_chart(dot_tree)

st.markdown("---")

st.write(':red-badge[Step 5:] **Observe model performnace feedback/evaluation metrics:**')

# show model performance feedback for linear regression
if 'mse' in st.session_state and 'rmse' in st.session_state and 'r2' in st.session_state:
    st.markdown('##### Performance Metrics')
    st.write(f":green-background[**Mean Squared Error (MSE):**] {st.session_state.mse:.3f}")
    st.write(f":violet-background[**Root Mean Squared Error (RMSE):**] {st.session_state.rmse:.3f}")
    st.write(f':red-background[**R-squared (R² Score):**] {st.session_state.r2:.3f}')
    st.caption('''
               :green[MSE] measures the average of the squared differences between the predicted and actual value of the data.

               :violet[RMSE] is the root of the MSE, measuring average distance between predicted and actual in a model.

               :red[R² Score], also known as the coefficient of determination, represents the proportion of the variance of the 
               target that can be explained by the features.
               ''')

# show model performance feedback for logistic regression
if 'logreg_accuracy' in st.session_state:
# generate confusion matrix
    st.markdown('##### Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(st.session_state.logreg_cm, annot = True, cmap = 'Greens')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    st.caption('The :green[***confusion matrix***] is a chart of four boxes displaying the number of correct (top-left and bottom-right) \
               and incorrect (top-right and bottom-left) predictions.')
# display classification report
    st.markdown('##### Classification Report')
    report_df = pd.DataFrame(st.session_state.logreg_report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Reds'))
    st.caption('''
               :red[***Accuracy***] is the overall percentage of correct classifications. 
               
               :red[***Precision***] is the positive predictive value, or the percentage of data points that were correctly predicted positive. 
               
              :red[ ***Recall***] is the true positive rate, or the portion of actually positive data points that were  that were also 
               predicted positive. 
               
               :red[***F1-Score***] is the balanced mean of precision and recall.
               ''')
    
# illustrate ROC and AUC
    st.markdown('##### ROC Curve')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(st.session_state.logreg_fpr, st.session_state.logreg_tpr, lw=2, label=f'ROC Curve (AUC = {st.session_state.logreg_roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    st.caption('''
               The :blue[***ROC Curve***], which stands for receiver operating characteristics, is a graph showing the true positive rate \
               vs. the false positive rate at various thresholds.' 
               'The more the curve aligns with the :orange[random guess] line, the closer the predictions are to being as good as \
               randomly guessing.
               
               :blue[***AUC***], or area under the curve, summarizes the overall model performance into a single metric, with 1.0 representing \
               a perfect delineation and 0.5 indicating the model is equivalent to random guessing.''')

# show model performance feedback for decision tree
if 'dt_accuracy' in st.session_state:
# generate confusion matrix
    st.markdown('##### Confusion Matrix')
    fig, ax = plt.subplots()
    sns.heatmap(st.session_state.dt_cm, annot = True, fmt='d', cmap = 'Greens')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    st.caption('The :green[***confusion matrix***] is a chart of four boxes displaying the number of correct (top-left and bottom-right) \
               and incorrect (top-right and bottom-left) predictions.')
# display classification report
    st.markdown('##### Classification Report')
    report_df = pd.DataFrame(st.session_state.dt_report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Reds'))
    st.caption('''
               :red[***Accuracy***] is the overall percentage of correct classifications. 
               
               :red[***Precision***] is the positive predictive value, or the percentage of data points that were correctly predicted positive. 
               
               :red[***Recall***] is the true positive rate, or the portion of actually positive data points that were  that were also 
               predicted positive. 
               
               :red[***F1-Score***] is the balanced mean of precision and recall.
               ''')
    
# illustrate ROC and AUC
    st.markdown('##### ROC Curve')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(st.session_state.dt_fpr, st.session_state.dt_tpr, lw=2, label=f'ROC Curve (AUC = {st.session_state.dt_roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    st.caption('''
               The :blue[***ROC Curve***], which stands for **receiver operating characteristics**, is a graph showing the true positive rate \
               vs. the false positive rate at various thresholds.' \
               'The more the curve aligns with the :orange[random guess] line, the closer the predictions are to being as good as \
               randomly guessing.
               
               :blue[***AUC***], or **area under the curve**, summarizes the overall model performance into a single metric, with 1.0 representing \
               a perfect delineation and 0.5 indicating the model is equivalent to random guessing.''')

# fix text input for decision tree max depth
# maybe add max leaf nodes? not necessary though