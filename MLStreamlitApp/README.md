## Interactive Machine Learning App

### ☀️ Welcome to my *"Interactive Machine Learning App"* created to allow users to run supervised machine learning models on a dataset of their choice and evaluate the quality of the model through multiple performance feedback measurements. :sunny: 
---
#### But before we dive right in :diving_mask: ... Let's get familiar with the terminology 😎

:rose: What is __supervised machine learning__?
> This is a type of machine learning where algorithms use **labeled data** to make a prediction or a decision. This means that each of the features have their unique names and that the target feature, which is what the algorithm is attempting to predict, is known.
> This method finds the proper formula or approach for predicting or classifying unknown data points.
#### :cactus: Models in the App: :cactus:
:seedling: What is __linear regression__?
> Linear regression is a model that uses the linear regression equation to determine the coefficients of each feature, which measures how much each feature has an impact on the outcome of the data. As can be seen by the equation below, the coefficients are represented by the greek letter beta, $\beta$. This model works for predicting continuous outcomes, such as the price of houses, the probability of recividism, etc.
> > $\{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$

:palm_tree: What is __logistic regression__?
> Logistic regression uses the logistic regression equation to determine the appropriate coefficient for each feature. These coefficients as well are represented by beta, $\beta$. Unlike linear regression, this model is applied to datasets with binary outcomes, such as yes or no, raining or not raining, and spam or ham (not spam) and thus cannot be used to predict continuous numbers. 
> > $\{y} = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n)}}$

:deciduous_tree: What is the __decision tree__?
> The decision tree is a classification model which asks yes/no questions to divide the data into groups with similar features. Instead of assigning coefficients to the features, this model uses the features to ask appropriate questions and group data based on the questions. Although decision trees aren't limited to binary outcomes, the model in this app will only be able to derive two.
>
> For this model, you may choose the depth, which is the amount of times the data is split up by questions, which is also represented by the number of levels in the tree. Setting a number for this will prevent outrageous and meaningless splitting and increase interpretability for users. The larger the depth, the more specific the data groups become.
> > <image src="https://blog.mindmanager.com/wp-content/uploads/2022/03/Decision-Tree-Diagram-Example-MindManager-Blog.png" alt="image" width="300"/>
---
Now that we understand the tools we are using, let's step into the process!🌟
#### :watermelon: Prep kit! :watermelon:
- Streamlit Community Cloud account https://share.streamlit.io/
- A `.csv` dataset *(optional)*
- An environment that can read .py files and run streamlit (ex. Visual Studio Code <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" alt="image" width="21"/> )
- The following libraries:
  -  graphviz (ver. 0.20.3)
  -  matplotlib (ver. 3.8.4)
  -  numpy (ver. 2.2.4)
  -  pandas (ver. 2.2.3)
  -  sckikit_learn (ver. 1.4.2)
  -  seaborn (ver. 0.13.2)
  -  streamlit (ver. 1.44.1)
- If you wish to run the app without using an enviroment, you can access the deployed version here: [Interactive Machine Learning App](https://kang-data-science-portfolio-7czws8sggmdaop3hcqzrcq.streamlit.app/)

#### :whale2: Running the App :whale2:
1. Download the project's script, *interactive_ml_app.py*, and requirement file, *requirements.txt*, onto your device by clicking the download icon near the top right corner of the screen <image src="https://static.vecteezy.com/system/resources/previews/019/879/209/non_2x/download-button-on-transparent-background-free-png.png" alt="image" width="21"/>
2. Open File Explorer (Windows) <image src="https://static.wikia.nocookie.net/windows/images/0/04/File_Explorer_Icon.png/revision/latest?cb=20240208004644" alt="image" width="21"/> or Finder (Mac) <image src="https://upload.wikimedia.org/wikipedia/commons/c/c9/Finder_Icon_macOS_Big_Sur.png" alt="image" width="21"/>
3. Locate both files (the requirements and the script) and place them into a new folder named appropriately together (ex. InteractiveMLApp)
4. Open the folder onto your environment
5. Open the terminal and enter `cd {folder name}` to ensure you are in the right folder
7. Type `streamlit run interacitive_ml_app.py` into the terminal and press enter
8. A website for the app should pop up on your browser, ready to go! 👍

#### :cake: Notes for Using the App :cake:
- Please make sure that your custom file is a `.csv` file under 200MB and not in any other file form
- If an error pops up while trying to run a model, it may be that the model does not support the type of data or target in the fed dataset
  - ex: the logistic regression model cannot take datasets where the target is a continuous ouput
- The app is designed so that the user can run another model on the same dataset without having to reinsert the dataset or the features/target by simply clicking on another model button
- Subsequent steps will produce errors if preceding steps have not been carried out properly
- The maximum depth for the decision three is 20
---
#### Understanding the Performance Feedback :ocean: [^1]

Name | Description
-------------| -------------
__Regression Evaluation Metrics__ | _MSE_, _RMSE_, and $R^2$ are tools used to evaluate the performance of a continuous value predictor. The first two focus on the difference between the actual and predicted value, while the last metric quantifies the amount of the dependent variable's variance that can be explained by the model's independent variable.
__Confusion Matrix__  | Four-boxed graph representing whether data points were true positives, false positives, true negatives, or false negatives. Higher numbers for true positives and true negatives are preferable, since it means the model was able to accurately distribute more data into their appropriate binary outcome.
__Classification Report__ | Chart displaying metrics such as _accuracy_, _precision_, _recall_, and _F1-score_. Higher values for any of these are preferable, but focus on a certain metric depends on whether false positives for false negatives are costly.
__ROC Curve & AUC__ | A curve drawn by putting the model's true positive rate against its false positive rate. The area under the curve of this graph, called AUC, ranges from 0.5 to 1.0, with 1.0 being perfect and 0.5 meaning that the model is not better at predicting the correct outcome than would be random guessing.
[^1]: You can find more detailed descriptions on the app itself
---
#### Tools & References :penguin:

This app was created through VSCode  <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" alt="image" width="19"/>  and Streamlit <image src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGDKmSgL7UJ6sstMUQTtjI2iDN7ClN2jRZ5Q&s" alt="image" width="21"/>

Here are the informational references used:
- [Grokking Machine Learning by Manning Editors](https://github.com/luisguiserrano/manning)
- [GeeksforGeeks](https://www.geeksforgeeks.org/regression-metrics/)
- [Scikit-learn](https://scikit-learn.org/stable/modules/tree.html)
---
Thank you for visiting my app and I hope you had an informational, useful experience with supervised machine learning! 💙💛


