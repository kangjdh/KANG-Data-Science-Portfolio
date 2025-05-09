# Interactive Unsupervised Machine Learning App

## Welcome fellow data explorers! :smile: 

### This app is designed to allow users to input a dataset of choice (or choose from included examples) and run different unsupervised machine learning techniques to discover meaningful insight within the data. 
---
### First let's introduce ourselves to the technical language! 

:star: What is ___unsupervised machine learning___? :star:

Unsupervised machine learning, as compared with supervised machine learning, is a type of artificial intelligence that aims at handling _unlabeled data_, meaning the target variable is unknown or does not exist. Rather than making predictions, these techniques aim to identify structure, group observations, reduce noise, or uncover hidden patterns.

### :watermelon: Models in the App: :watermelon:
:melon: What is ___Principal Component Analysis___?

Principal component analysis, more commonly known as PCA, is a method of reducing dimensionality while caputuring maximum variance using principal components. This simplifies complex data by explaining each point with a certain number of most determinant factors. 
> **Benefits:** improves computational efficiency, enhances data visualization, and prepares data for better model performance.

:pineapple: What is ___K-Means Clustering___?

KMeans groups data into k-clusters and find the optimal centroid for each centroid to discover hidden structures in unlabeled data, segment large datasets into meaningful subgroups, and identify patterns or outliers for further analysis or preprocessing, including data imputation.
> **Benefits:** fast and simple to implement, scales well to large datasets, intuitive and easy to interpret, and useful for data exploration, customer segmentation, etc.

:peach: What is the ___Hierarhcial Clustering___?

Builds an agglomerative (bottom-up) tree, called a _dendrogram_, of nested clusters that remain in a group until proven otherwise to uncover multi‑level structure in unlabeled data, segment datasets into variable‑sized groups, and also detect patterns or outliers for further analysis or preprocessing.
> **Benefits:** can choose number of clusters after seeing dendrogram, works with many distance metrics / linkage criteria, provides full merge history for exploratory insight, and is valuable for gene‑expression studies, market segmentation, text/topic grouping, etc


---
Now let's get to the fun part!

### :gift: Prep kit! :gift:
- Streamlit Community Cloud account https://share.streamlit.io/
- A `.csv` dataset *(optional)*
- An environment that can read .py files and run streamlit (ex. Visual Studio Code <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" alt="image" width="21"/> )
- The following libraries:
  -  kagglehub (ver. 0.3.12)
  -  matplotlib (ver. 3.8.4)
  -  numpy (ver. 2.2.5)
  -  pandas (ver. 2.2.3)
  -  sckikit_learn (ver. 1.4.2)
  -  scipy (ver. 1.15.2)
  -  seaborn (ver. 0.13.2)
  -  streamlit (ver. 1.44.1)
- If you wish to run the app without using an enviroment, you can access the deployed version here: [Interactive Unsupervised Machine Learning App](https://kang-data-science-portfolio-7czws8sggmdaop3hcqzrcq.streamlit.app/)

### :jigsaw: Running the App :jigsaw:
1. Download the project's script, *unsupervised_ml_app.py*, and requirement file, *requirements.txt*, onto your device by clicking the download icon near the top right corner of the screen <image src="https://static.vecteezy.com/system/resources/previews/019/879/209/non_2x/download-button-on-transparent-background-free-png.png" alt="image" width="21"/>
2. Open File Explorer (Windows) <image src="https://static.wikia.nocookie.net/windows/images/0/04/File_Explorer_Icon.png/revision/latest?cb=20240208004644" alt="image" width="21"/> or Finder (Mac) <image src="https://upload.wikimedia.org/wikipedia/commons/c/c9/Finder_Icon_macOS_Big_Sur.png" alt="image" width="21"/>
3. Locate both files (the requirements and the script) and place them into a new folder named appropriately together (ex. UnsupervisedMLApp)
4. Open the folder onto your environment
5. Open the terminal and enter `cd {folder name}` to ensure you are in the right folder
7. Type `streamlit run unsupervised_ml_app.py` into the terminal and press enter
8. A website for the app should pop up on your browser, and you are now ready to go! :tada:

### :zap: Notes for Using the App :zap:
- Please make sure that your custom file is a `.csv` file under 200MB and not in any other file form
- If an error pops up while trying to run a model, it may be that the model does not support the type of data or target in the fed dataset
- The app is designed so that the user can run another model on the same dataset without having to reinsert the dataset or  features by simply clicking on another model button
- Subsequent steps will produce errors if preceding steps have not been carried out properly
- The minimum number of components or clusters that will allow all models to run is 2
- You may use labeled data as well to run the models
---
### Understanding the Performance Feedback :ocean: [^1]

Name | Description
-------------| -------------
__PCA Scatter Plot__ | _Scatter plot_ that plots the data points in a coordinate system using the first two principal components as the axes. This visualizes how the data is spread out and whether distinct groups exist.
__Scree Plot__ | A graph showing _cumulative variance explained_ vs. number of components. This can be used to decide the optimal number of components needed to capture most of the variance in the data.
__Classification Report__ | Chart displaying metrics such as _accuracy_, _precision_, _recall_, and _F1-score_. Higher values for any of these are preferable, but focus on a certain metric depends on whether false positives for false negatives are costly.
__Elbow Method__ | Graph plotting the _within-cluster sum of squares (WCSS)_ against different values of k. The _elbow point_ is where the rate of decrease changes sharply, indicating an optimal k.
__Silhouette Score__ | Number quantifying how similar an object it in relation to its own cluster compared to other clusters, with higher numbers being preferable. The value of k with the highest average silhouette score is the optimal k.
__Dendrogram__ | Hierarchial tree that uses _ward linkage_, merging clusters that yield smallest increase in within-cluster variance. This represents the similarity structure by showing who merged early and reasonable cut heights (horizontal line) for k clusters. 

[^1]: You can find more detailed descriptions on the app itself
---
### Tools & References :penguin:

This app was created through VSCode  <image src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Visual_Studio_Code_1.35_icon.svg/2048px-Visual_Studio_Code_1.35_icon.svg.png" alt="image" width="19"/>  and Streamlit <image src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGDKmSgL7UJ6sstMUQTtjI2iDN7ClN2jRZ5Q&s" alt="image" width="21"/>

Here are the informational references used:
- [Grokking Machine Learning by Manning Editors](https://github.com/luisguiserrano/manning)
- [GeeksforGeeks](https://www.geeksforgeeks.org/unsupervised-learning/)
- [Scikit-Learn](https://www.learndatasci.com/glossary/hierarchical-clustering/)
---
Thank you for visiting my app and I hope you had an informational, useful experience with unsupervised machine learning! 💙💛
