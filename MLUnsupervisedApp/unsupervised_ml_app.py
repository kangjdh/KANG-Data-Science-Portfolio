import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# title of the app
st.title('Interactive Unsupervised Machine Learning App!')

# subheader / explanation for the app's purpose and functions
st.subheader(':star2: Welcome to my :rainbow[unsupervised machine learning] app! ')
st.text('Enter your dataset (or choose from a sample) and run an unsupervised' \
' machine learning model of your choice to find out the secrets hidden between the mass of numbers.')

st.divider()

## data uploading section

# title the section
st.subheader(":shaved_ice: **Upload Your Data**", divider=True)

# define dataframe before addition of data
# this step is added to preserve the chosen dataset between reruns using different machine learning models
if 'df' not in st.session_state:
    st.session_state.df = None

# add a helper function so entire code can be reset when new model is chosen and old variables are thrown out
def reset_model_states(new_df):
    if 'df' not in st.session_state or not new_df.equals(st.session_state.df):
        for key in ["labels", "clustered_dataset", "cluster_labels", "X_scaled", "pca_xpca", "kmeans_clusters", "pca"]:
            st.session_state.pop(key, None)
        st.session_state.df = new_df
    # create a variable out of first non-numeric column of the dataset to use it as labels for graphs, etc.
    non_numeric_cols = new_df.select_dtypes(exclude=[np.number]).columns
    # identify the target variable if there is a column with such name in dataframe
    if 'target' in new_df.columns:
        # save target for future use
        st.session_state.y = new_df['target'].values
        st.session_state.target_names = new_df['target'].unique()
    # use first non_numeric column name as target, even if not named target explicity
    elif len(non_numeric_cols) > 0:
        st.session_state.y = pd.factorize(new_df[non_numeric_cols[0]])[0]  # encode first non-numeric col
        st.session_state.target_names = new_df[non_numeric_cols[0]].unique()
    # if neither available, there will be no target identified and remain as None
    else:
        st.session_state.y = None
        st.session_state.target_names = None
    st.session_state.non_numeric_cols = non_numeric_cols

# define variable custom dataset
# add a file uploader to allow users to insert their custom dataset
custom_dataset = st.file_uploader(':violet-badge[Step 1:] **Upload your .csv dataset file here**:', type=['csv'])
# label the custom dataset as the df
if custom_dataset is not None:
    new_df = pd.read_csv(custom_dataset)
    # this is for the purpose of reseting the model when rerunning a new model with the same dataset
    reset_model_states(new_df)

# include dataset options that can be used in the case user does not have custom dataset
st.write('Example datasets!:')

# define a helper function to clear the session states from previously run models called clear_model_states
# this will get rid of all graphs from previous runs when clicking on a new model
def clear_model_states(exclude=None):
    model_keys = [
        'pca_num_comp', 'pca_explvar', 'pca_explvar_cumsum', 'pca_xpca', 'pca_xstd', 'pca',
        'kmeans_clusters', 'kmeans', 'xpca',
        'clustered_dataset', 'cluster_labels', 'X_scaled'
    ]
    # the following keys will be deleted if they are not in the excluded section for deletion
    for key in model_keys:
        if exclude is None or key not in exclude:
            st.session_state.pop(key, None)

# set up columns for the  sample datasets to display all of the options in one line
col1, col2 = st.columns(2)
# add a button for the 'California Housing' dataset for column 1
with col1:
    if st.button('Breast Cancer', type='primary'):
        # load dataset from sklearn.datasets
        from sklearn.datasets import load_breast_cancer
        # set dataset as the df
        data = load_breast_cancer()
        # define the feature matrix and target of the breast cancer dataset
        X = data.data
        y = data.target
        # label the feature and target names
        #feature_names = data.feature_names
        target_names = data.target_names
        # save the df as st.session_state to preserve information for potential reruns
        new_df = pd.DataFrame(X, columns=data.feature_names)
        reset_model_states(new_df)
        #st.session_state.feature_names = feature_names
        st.session_state.target_names = target_names
        st.session_state.y = y
# add a button for the 'Country-Level Indicator' dataset for column 2
with col2:
    if st.button('Country Level', type='primary'):
        # download latest version of the dataset from kaggle
        import kagglehub
        path = kagglehub.dataset_download("rohan0301/unsupervised-learning-on-country-data")
        # assuming that the file is named 'Country-data.csv' and is inside downloaded directory
        import os
        file_path = os.path.join(path, 'Country-data.csv')
        # set dataset as the df
        data = pd.read_csv(file_path)
        # save the dataset as st.session_state to preserve information for potential reruns
        new_df = data  # or the country dataset DataFrame
        reset_model_states(new_df)

# rename st.session_state.df to dataset for simplification of further coding
dataset = st.session_state.df

# create a button that allows users to preview the first couple of lines of their dataset
if dataset is not None:
    feature_names = dataset.select_dtypes(include=[np.number]).columns.tolist()
    st.session_state.feature_names = feature_names
    if st.button('Dataset Preview'):
        st.dataframe(dataset.head())
        # provide information on the shape of the dataset (total rows x total columns)
        st.write(dataset.shape)
        # describe what the shape means for users
        st.caption('The numbers within the parantheses above indicates number of rows' \
        ' and number of columns in the dataset, respectively')

## unsupervised machine learning model selection section

# title the section
st.subheader(":watermelon: **Choose a Unsupervised Machine Learning Model**", divider='green')
# describe the purpose of this section
st.write('Following uploading the dataset, the next step is to run an unsupervised machine learning' \
' model on the chosen dataset... ')

#
st.markdown(' :blue-badge[Step 2:] **Click on a tab below to choose a model**')
  
# create tabs for the unsupervised machine learning models
tab1, tab2, tab3 = st.tabs(['Principal Component Analysis', 'KMeans Clustering', 'Hierarchial Clustering'])

# add an option for Principal Component Analysis (PCA) into tab 1
with tab1:
    # download additonal necessary libraries needed to run PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    # explain the purpose of the model, how it works, and its benefits
    st.write(':violet[***Principal Component Analysis***], also known as ***PCA***, is a method of unsupervised machine ' \
    'learning that simplifies complex, high-dimensional datasets by reducing the number of components while capturing ' \
    'maximum variance using the principal components.')
    st.write('This **dimensionality reduction** technique increases computational efficiency, enhances data' \
    ' visualization, and prepares data for better model performance.')
    # extract only numeric columns to use as max number of components user can try for
    numeric_df = dataset.select_dtypes(include=['number'])
    max_comps = numeric_df.shape[1]
    # allow users to adjust the number of components they want
    num_components = st.number_input(':green-badge[Step 3:] **Enter the desired number of components:** ', min_value=2, max_value=max_comps)
    # center and scale the features
    # this step is especially important for PCA since this model is responsive to variable scales 
    scaler = StandardScaler()
    if dataset is not None:
        X_std = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))
    # create a button to run the PCA model against the dataset using the PCA library
    st.markdown(':orange-badge[Step 4:] **Click on the button to run the model!**')
    if st.button('Run PCA', type='primary'):
        clear_model_states(exclude=['y', 'feature_names', 'target_names'])
        pca = PCA(n_components = num_components)
        X_pca = pca.fit_transform(X_std)
        # display message that lets users know the model was run successfully
        st.success(':white_check_mark: PCA model ran successfully!:white_check_mark:')
        # calculate the Explained Variance Ratio, or, the proportion of variance explained by each component
        explained_variance = pca.explained_variance_ratio_
        np.cumsum(explained_variance)
        # store calculations into st.session to use values when making calculations outside of current loop
        st.session_state.pca_num_comp = num_components
        st.session_state.pca_explvar = explained_variance
        st.session_state.pca_explvar_cumsum = np.cumsum(explained_variance)
        st.session_state.pca_xpca = X_pca
        st.session_state.pca_xstd = X_std
        st.session_state.pca = pca

# add an option for KMeans Clustering into tab 2
with tab2:
    # download additonal necessary libraries needed to run KMeans Clustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    # explain the purpose of the model, how it works, and its benefits
    st.write(':violet[***KMeans Clustering***] is a method of grouping data points in _k-clusters_ and finding' \
    ' the optimal centroid for each cluster to discover hidden structures in unlabled data and segment ' \
    'large datasets into meaningful subgroups.')
    st.write('This technique is fast and easy to implement, scales well to large datasets, is intuitive ' \
    'and easy to interpret, and is useful for data exploration, customer segmentation, etc.')
    # allow users to initialize by selecting a number for "k"
    k_clust = st.number_input(':green-badge[Step 3:] **Enter the number of clusters desired:** ', min_value=2, max_value=25)
    # center and scale the features
    # this step is important for KMeans since it relies on distance calculations and can be 
    # biased by the scale of features 
    scaler = StandardScaler()
    if dataset is not None:
        X_std = scaler.fit_transform(dataset.select_dtypes(include=[np.number]))
        # create a button to run the KMeans model against the dataset using the Kmeans library
        st.markdown(':orange-badge[Step 4:] **Click on the button to run the model!**')
        if st.button('Run KMeans Clustering', type='primary'):
            clear_model_states(exclude=['y', 'feature_names', 'target_names'])
            kmeans = KMeans(n_clusters = k_clust, random_state = 42)
            clusters = kmeans.fit_predict(X_std)
            # run a PCA for scatter plot in performance feedback section
            pca = PCA(n_components = k_clust)
            X_pca = pca.fit_transform(X_std)
            # display message that lets users know the model was run successfully
            st.success(':white_check_mark: KMeans model ran successfully!:white_check_mark:')
            # store calculations into st.session to use values when making calculations outside of current loop
            st.session_state.kmeans_clusters = clusters
            st.session_state.kmeans = kmeans
            st.session_state.xpca = X_pca

# add an option for hierarchial clustering into tab 3
with tab3:
    # download additonal necessary libraries needed to run hierarchial clustering
    from sklearn.preprocessing import StandardScaler
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.cluster import AgglomerativeClustering
    # explain hierarcial clustering
    st.write(':violet[***Hierarchial Clustering***] is a method of clustering data based on similarity ' \
    ' creating a tree like structure called a _dendrogram_. Each data point starts as its own cluster at the bottom and is' \
    ' progressively merged or split with other data points based on similarity.')
    st.write('With this technique you can choose the number of clusters after seeing the dendrogram, and it works with' \
    ' many distance metrics/linkage criteria, provides full merge history for exploratory insight, and is valuable for ' \
    'gene-expression studies, market segmentation, text/topic grouping, etc.')
    # create a dataframe of just numeric columns since only the features will be preserved from the dataset
    features_df = dataset.select_dtypes(include = [np.number])
    # in the case that there are more than one non-numeric column, only preserve the first
    non_numeric_cols = st.session_state.non_numeric_cols
    if len(non_numeric_cols) > 0:
        label_column = non_numeric_cols[0]
        # create a list out of the variables within the first column
        labels = dataset.loc[features_df.index, label_column].tolist()
        # save labels to st.session_state so it can be used outside of this loop and in codes for graphs, etc.
        st.session_state.labels = labels
    # if no non-numeric values have been found, consider the labels as nonexistent
    else:
        labels = None
    # explain the purpose of a santiy-check visual and broadly how it works
    st.write('A **sanity check visual** makes it easy to confirm that there are no obvious errors or anomalies in the data'
    ' being used that must be addressed.')
    st.write('Below is a button that provides a histogram representing the distribution of each '
    'feature that is being used in the model.')
    # make a button for users to optionally view a sanity-check visual
    if st.button('Feature Distribution', type='secondary'):
        features_df.hist(figsize=(12,8), edgecolor="k", bins=15)
        plt.suptitle("Distribution of each numeric feature", y=1.02)
        plt.tight_layout()
        st.pyplot(plt)
    # because hierarchial clustering uses Euclidean distances, indicators measured on bigger scales would
    # have a bigger impact, so the features must be standardized to zero mean
    # take only the numeric columns since only features should be preserved in the dataset
    scaler = StandardScaler()
    if dataset is not None:
        X_scaled = scaler.fit_transform(features_df)
    # prepare information needed to print a dendrogram
    # explain the purpose of a dendrogram and broadly how it works
    st.write('A **dendrogram** is a visual representation showing how data points or groups are related based on '
    'their similarity. This shows similarity structure and reasonable cut heights for k clusters.')
    # Standardize the numeric features (centering and scaling)
    Z = linkage(X_scaled, method="ward")
    # print a dendrogram
    # this chart gives insight about similarity structures (who merges early) and reasonable cut heights 
    # for k clusters, truncating to the last 30 to make the graph readable
    # add a checkbox for toggling truncated view
    truncated_view = st.checkbox("Show Truncated Version (Last Merges Only)")
    # start a fresh plot with selected figure size
    plt.figure(figsize=(20, 7))
    # draw the appropriate dendrogram based on user selection
    if truncated_view:
        dendrogram(Z,
                truncate_mode="lastp",  # this shows only the last merging
                labels=labels)
        plt.title("Hierarchical Clustering Dendrogram (Truncated)")
    else:
        dendrogram(Z,
                labels=labels)
        plt.title("Hierarchical Clustering Dendrogram (Full)")
    # add labels and print plot
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    st.pyplot(plt)
    # clear figure to avoid overlap when figure is redrawn
    plt.clf()
    # make a sign informing users of the use of ward linkage in this hierarchial clustering method
    st.info('This model uses **ward linkage** (merging clusters yielding *smallest* increase in total within-cluster' \
    ' variance))')
    # let users decide the number of k 
    k = st.number_input(':green-badge[Step 3:] **Enter the number of clusters desired:** ', min_value=1, max_value=25)
    # create a button to run the Hierarhcial model against the dataset using the hierarchy library
    st.markdown(':orange-badge[Step 4:] **Click on the button to run the model!**')
    if st.button('Run Hierarchial Clustering', type='primary'):
        clear_model_states(exclude=['y', 'feature_names', 'target_names'])
        # assign cluster label with fit_predict()
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        cluster_labels = agg.fit_predict(X_scaled)
        # assign clusters to a copy of dataset
        clustered_dataset = dataset.copy()
        clustered_dataset['Cluster'] = cluster_labels
        # return a copy of cluster labels in the form of an array
        cluster_labels = clustered_dataset["Cluster"].to_list()
        # display message that lets users know the model was run successfully
        st.success(':white_check_mark: Hierarchial clustering model ran successfully!:white_check_mark:')
        # store calculations into st.session to use values when making calculations outside of current loop
        st.session_state.clustered_dataset = clustered_dataset
        st.session_state.cluster_labels = cluster_labels
        st.session_state.X_scaled = X_scaled


st.subheader(":tangerine: **Review Results & Performance Feedback**", divider='orange')

## print model results for each model
st.markdown(':red-badge[Step 5:] **Observe the results derived from the model**')

# PCA

# make sure this section runs only if PCA has been run
if 'pca_xpca' in st.session_state and 'pca_explvar_cumsum' in st.session_state:
    X_pca = st.session_state.pca_xpca

    # display explained variance ratio (the proportion of variance explained by each component)
    st.write(":violet[**Cumulative Variance Explained:**]", st.session_state.pca_explvar_cumsum)
    st.caption(':violet[**Cumulative variance explained**] is how much of the variance in the data' \
    ' can be attributed to the most determinant chosen number of components. The first row is ' \
    'variance explained by the first component, the second row representing that of the first two components, etc.')
    feature_names = st.session_state.feature_names
    pca = st.session_state.pca
    st.write(':blue[**PCA Scatter Plot**]')
    # generate a scatter plot of PCA Scores
    # this one is to include target names if they exist
    if 'target_names' in st.session_state and 'y' in st.session_state:
        target_names = st.session_state.target_names
        y = st.session_state.y
        # identify how many unique target names exist
        n_clusters = len(target_names)
        plt.figure(figsize=(8, 6))
        # for aesthetic purposes, create a list of colors to associate with 20 or less clusters
        pretty_colors = ['turquoise', 'hotpink', 'lawngreen', 'violet', 'lightcoral', 'limegreen', 'turquoise', 'palegreen',
                  'chartreuse','plum','lightpink', 'deepskyblue', 'deeppink', 'darkorchid', 'powderblue', 'orange', 
                  'turquoise', 'springgreen', 'mediumspringgreen', 'lightcyan']
        if n_clusters <= len(pretty_colors):
            colors = pretty_colors
        # if there are more than 20 colors needed, this color palette will be the alternative
        else:
            colors = sns.color_palette("hls", n_clusters)
        for i, (color, target_name) in enumerate(zip(colors, target_names)):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.7,
                        label=target_name, edgecolor='gainsboro', s=60)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA: 2D Projection')
        plt.legend(loc='best')
        plt.grid(True)
        st.pyplot(plt)
        # explain below the graph what a PCA graphs shows and the purpose of it
        st.caption('The visualization above graphs the data points in relation to the first' \
        ' two principal components. This makes spread and groups easier to see.')
    else:
        # this graphs the same PCA plot but for datasets that do not have a target
        plt.figure(figsize=(8, 6))
        # here you can see that the brackets following X_pca are filled with a colon instead
        plt.scatter(X_pca[:, 0], X_pca[:, 1], color='turquoise', alpha=0.7,
                    edgecolor='gainsboro', s=60)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA: 2D Projection (Unlabeled Data)')
        plt.legend(loc='best')
        plt.grid(True)
        st.pyplot(plt)
        # explain below the graph what a PCA graphs shows and the purpose of it
        st.caption('The visualization above graphs the data points in relation to the first' \
        ' two principal components. This makes spread and groups easier to see.')

    # create the combined scree plot and bar graph against each other to represent optimal number for k
    # generate a scree plot which displays the cumulative explained variance
    # this plot helps in determining how many components to retain (looking for the "elbow")
    n_comps = st.number_input('Enter the number of components to observe variance explained for:', 
                              min_value=1, max_value=X_std.shape[1])
    pca_full = PCA(n_components = n_comps).fit(X_std)
    # create a bar plot with each component's variance explained
    # define variables needed to calculate scree and bar plot
    explained = pca_full.explained_variance_ratio_ * 100  # individual variance (%) per component
    components = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)
    # graph bar plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_color = 'lightseagreen'
    ax1.bar(components, explained, color=bar_color, alpha=0.8, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Individual Variance Explained (%)', color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)
    ax1.set_xticklabels([f"PC{i}" for i in components])
    # add percentage labels on each bar
    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10, color='black')
    # create a second y-axis for cumulative variance explained
    ax2 = ax1.twinx()
    line_color = 'tomato'
    ax2.plot(components, cumulative, color=line_color, marker='o', label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Explained (%)', color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)
    # remove grid lines
    ax1.grid(False)
    ax2.grid(False)
    # Combine legends from both axes and position the legend inside the plot (middle right)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(0.85, 0.5))
    plt.title('PCA: Variance Explained', pad=20)
    plt.tight_layout()
    st.pyplot(plt)
    # explain the purpose of the graphs and what each represent
    st.caption('This graph shows a :red[scree plot], which graphs cumulative variance explained, against ' \
    'a :green[bar plot] representing variance explained by each individual component.')
    st.caption('The point on the plot before the added variance '
    'explained by the next component becomes relatively small increase indicates a good number to use as k.')

# KMean Clustering

# make sure this section runs only if KMeans has been run
if 'kmeans_clusters' in st.session_state: # and 'pca_xpca' in st.session_state:
    clusters = st.session_state.kmeans_clusters
    X_pca = st.session_state.xpca
    y = st.session_state.y

    # Create a scatter plot of the PCA-transformed data, colored by KMeans cluster labels
    plt.figure(figsize=(8, 6))
    # create a list of colors to use to assign to each cluster (up to 17 clusters) for aesthetic purposes
    pretty_colors = ['hotpink', 'skyblue','aquamarine','violet','palegreen','chartreuse','plum','lightpink', 'deepskyblue', 
              'deeppink', 'darkorchid', 'powderblue', 'orange', 'turquoise', 'springgreen', 'mediumspringgreen', 'lightcyan']
    num_clusters = len(np.unique(clusters))
    # choose colors depending on number of clusters
    if num_clusters <= len(pretty_colors):
        colors = pretty_colors
    else:
        colors = sns.color_palette("hls", num_clusters)
    # Iterate over unique cluster labels
    for cluster_label in np.unique(clusters):
        # Get indices of data points belonging to the current cluster
        indices = np.where(clusters == cluster_label)
        # Scatter plot for the current cluster using the corresponding color
        # and using the cluster_label as the legend label
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1],
                    color=colors[cluster_label], alpha=0.7, edgecolor='gainsboro', s=60, label=f'Cluster {cluster_label}')
    # label the graph
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(plt)
    # explain below the graph what a PCA graphs shows and the purpose of it
    st.caption('The visualization above graphs the data points, color-coded by the clusters as determined by k-means '
    'clustering, in relation to the first two principal components. This makes spread and groups easier to see.')

    # to assess how well the clusters match the true labels, create a classification report
    # display classification report
    if dataset is not None and 'kmeans_clusters' in st.session_state:
        # identify the target or y variable from dataset
        if 'target' in locals() or 'y' in locals():
            try:
                # download the necessary libraries
                from sklearn.metrics import classification_report
                # identify the true y and y to be predicted
                y_true = st.session_state.y
                y_pred = st.session_state.kmeans_clusters
                report = classification_report(y_true, y_pred, target_names=st.session_state.target_names, output_dict=True)
                # create a dataframe from the classifacation report to make it look more appealing
                report_df = pd.DataFrame(report).transpose()
                st.markdown('##### Classification Report')
                st.dataframe(report_df.style.background_gradient(cmap='Spectral'))
                # explain the important numbers in the classification report that users should understand
                st.caption('''
                            :green[***Accuracy***] is the overall percentage of correct classifications. 
                            
                            :red[***Precision***] is the positive predictive value, or the percentage of data points that were correctly predicted positive. 
                            
                            :blue[***Recall***] is the true positive rate, or the portion of actually positive data points that were  that were also 
                            predicted positive. 
                            
                            :red[***F1-Score***] is the balanced mean of precision and recall.
                            ''')
            # make exception warning appear if a classification report cannot be 
            # this will include if the number of targets identified by script above does not match the number of clusters requested by user
            except Exception as e:
                st.warning(f":warning: Classification report couldn't be generated: {e} :warning:")


    # to evaluate the best number of clusters, calculate using the elbow method and silhouette score
    # download necessary libraries
    from sklearn.metrics import silhouette_score
    # define the range of k values to try
    min_kval = int(st.number_input('Enter the minimum k value to try: ', min_value=2, step=1))
    max_kval = int(st.number_input('Enter the maximum k value to try: ', step=1, min_value=min_kval))
    ks = range(min_kval, max_kval+1)
    # starting from 2 clusters to 10 clusters
    wcss = []
    silhouette_scores = []
    # within-Cluster Sum of Squares for each k
    # silhouette scores for each k
    for k in ks:
        km = KMeans(k, random_state = 42)
        km.fit(X_std)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X_std, labels))
    # loop over the range of k values
    # create both results into the form of a chart
    # plot the Elbow Method result
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks, wcss, marker='o', color='orange')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    # plot the Silhouette Score result
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouette_scores, marker='o', color='mediumseagreen')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)
    # print both graphs side by side for easy visualization to the eye
    plt.tight_layout()
    st.pyplot(plt)
    # explain the elbow method
    st.caption(':orange[**The Elbow Method**] plots the within-cluster sum of squares (WCSS) against' \
    ' different values of k. The "elbow point," where the rate of decrease changes sharply, ' \
    'suggests an optimal k.')
    # explain the silhouette score
    st.caption(':green[**The Silhouette Score**] quantifies how similar a data point is to its own cluster'
    ' compared to other clusters. This graph computes the average silhouette score for different '
    'values of k, with the higher the score being the better.')

# Hierarchial Clustering

# make sure hierarchial clustering has been run before printing performance feedback
if 'clustered_dataset' in st.session_state and 'cluster_labels' in st.session_state and 'X_scaled' in st.session_state:
    cluster_labels = st.session_state.cluster_labels
    X_scaled = st.session_state.X_scaled
 
    # low dimensional insight using PCA
    # note: PCA is only for display and was not used to actually fit the clusters
    # Step 4: Visualize the Clustering Results Using PCA
    from sklearn.decomposition import PCA
    # Reduce the dimensions for visualization (2D scatter plot)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # plot the graph
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='winter', s=60, edgecolor='gainsboro', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Agglomerative Clustering on Data (via PCA)')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.grid(True)
    st.pyplot(plt)
    # explain below the graph what a PCA graphs shows and the purpose of it
    st.caption('The visualization above graphs the data points, color-coded by the clusters as determined by hierarchial'
    'clustering, in relation to the first two principal components. This makes spread and groups easier to see.')

    # create a silhouette elbow to find the optima value for k and its score
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # Range of candidate cluster counts
    min_kval = int(st.number_input('Enter the minimum k value to try: ', step=1, min_value=2))
    max_kval = int(st.number_input('Enter the maximum k value to try: ', step=1, min_value=min_kval))
    k_range = range(min_kval, max_kval+1)
    sil_scores = []
    for k in k_range:
        # Fit hierarchical clustering with Ward linkage (same as dendrogram)
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)
        # Silhouette: +1 = dense & well‑separated, 0 = overlapping, −1 = wrong clustering
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
    # Plot the curve
    plt.figure(figsize=(7,4))
    plt.plot(list(k_range), sil_scores, marker="o", color='mediumorchid')
    plt.xticks(list(k_range))
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Average Silhouette Score")
    plt.title("Silhouette Analysis for Agglomerative (Ward) Clustering")
    plt.grid(True, alpha=0.3)
    st.pyplot(plt)
    # print best k
    best_k = k_range[np.argmax(sil_scores)]
    st.write(f":violet[Best k by silhouette:] **{best_k}  (score={max(sil_scores):.3f})**")
    st.caption(':violet[**The Silhouette Score**] quantifies how similar a data point is to its own cluster'
    ' compared to other clusters. This graph computes the average silhouette score for different '
    'values of k, with the higher the score being the better.')




