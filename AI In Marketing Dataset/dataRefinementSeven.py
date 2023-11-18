# -*- coding: utf-8 -*-
"""
APPLY PRINCIPAL COMPONENT ANALYSIS AND ANALYZE THE RESULTS
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import plotly.express as px
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Now you can use pd.read_csv and other pandas functions
sales_df = pd.read_csv('/content/sales_data_sample.csv', encoding='unicode_escape')

sales_df

sales_df['COUNTRY'].value_counts().index

sales_df['COUNTRY'].value_counts()

# Function to add dummy variables to replace categorical variables

def dummies(x):
  dummy = pd.get_dummies(sales_df[x])
  sales_df.drop(columns = x , inplace = True)
  return pd.concat([sales_df, dummy], axis = 1)

sales_df = dummies('PRODUCTLINE')

sales_df

sales_df = dummies('DEALSIZE')
sales_df

y = pd.Categorical(sales_df['PRODUCTCODE'])
y

y = pd.Categorical(sales_df['PRODUCTCODE']).codes
y

# Since the number unique product code is 109, if we add one-hot variables, there
# would be additional 109 columns, we can avoid that by using categorical encoding
# This is not the optimal way of dealing with it but it's important to avoid curse of dimensionality
sales_df['PRODUCTCODE'] = pd.Categorical(sales_df['PRODUCTCODE']).codes

sales_df

# Reduce the original data to 3 dimensions using PCA for visualizig the clusters
pca = PCA(n_components = 3)
principal_comp = pca.fit_transform(sales_df_scaled)
principal_comp

pca_df = pd.DataFrame(data = principal_comp, columns = ['pca1', 'pca2', 'pca3'])
pca_df.head()

# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster':labels})], axis = 1)
pca_df

import plotly.express as px

# Assuming 'cluster' is a categorical column in your DataFrame
# If it's not categorical, you might need to convert it to a string or category
# pca_df['cluster'] = pca_df['cluster'].astype(str)

# Visualize clusters using 3D-Scatterplot
fig = px.scatter_3d(pca_df, x='pca1', y='pca2', z='pca3',
                    color='cluster', symbol='cluster', size_max=18, opacity=0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

import plotly.express as px
fig = px.scatter(pca_df, x='pca1', y='pca2', color='cluster', symbol='cluster', opacity=0.7)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()