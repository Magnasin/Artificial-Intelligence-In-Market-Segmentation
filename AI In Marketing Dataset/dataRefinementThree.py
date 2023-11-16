import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from IPython.display import display
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

sales_df = pd.read_csv('AI In Marketing Dataset\sales_data_sample.csv', encoding = 'unicode_escape')

sales_df_group = sales_df.groupby(by = "ORDERDATE").sum()
sales_df_group

# Convert ORDERDATE to datetime format
sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'])

# Extract month from ORDERDATE and create 'Month' column
sales_df['Month'] = sales_df['ORDERDATE'].dt.month_name()

# Group by month and sum the sales
sales_by_month = sales_df.groupby('Month')['SALES'].sum().reset_index()

# Visualize the sales by month
plt.figure(figsize=(10, 6))
sns.barplot(x='Month', y='SALES', data=sales_by_month, palette='viridis')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# Interactive plot using Plotly Express
fig = px.bar(sales_by_month, x='Month', y='SALES', labels={'SALES': 'Total Sales'})
fig.update_layout(title='Monthly Sales', xaxis_title='Month', yaxis_title='Total Sales')
fig.show()


"""MINI CHALLENGE #7:
- Plot the correlation matrix between variables
- Comment on the matrix results.
"""

plt.figure(figsize = (20, 20))
corr_matrix = sales_df.iloc[:, :10].corr()
sns.heatmap(corr_matrix, annot = True, cbar = False)

# It looks like the Quarter ID and the monthly IDs are highly correlated
# Let's drop 'QTR_ID' (or 'MONTH_ID')
sales_df.drop("QTR_ID", axis = 1, inplace = True)
sales_df.shape

# Let's plot distplots
# Distplot shows the (1) histogram, (2) kde plot and (3) rug plot.
# (1) Histogram: it's a graphical display of data using bars with various heights. Each bar groups numbers into ranges and taller bars show that more data falls in that range.
# (2) Kde Plot: Kernel Density Estimate is used for visualizing the Probability Density of a continuous variable.
# (3) Rug plot: plot of data for a single quantitative variable, displayed as marks along an axis (one-dimensional scatter plot).

import plotly.figure_factory as ff

plt.figure(figsize = (10, 10))

for i in range(8):
  if sales_df.columns[i] != 'ORDERLINENUMBER':
    fig = ff.create_distplot([sales_df[sales_df.columns[i]].apply(lambda x: float(x))], ['distplot'])
    fig.update_layout(title_text = sales_df.columns[i])
    fig.show()

# Visualize the relationship between variables using pairplots
plt.figure(figsize = (15, 15))

fig = px.scatter_matrix(sales_df,
    dimensions = sales_df.columns[:8], color = 'MONTH_ID')

fig.update_layout(
    title = 'Sales Data',
    width = 1100,
    height = 1100,
)
fig.show()

# A trend exists between 'SALES' and 'QUANTITYORDERED'
# A trend exists between 'MSRP' and 'PRICEEACH'
# A trend exists between 'PRICEEACH' and 'SALES'
# It seems that sales growth exists as we move from 2013 to 2014 to 2015 ('SALES' vs. 'YEAR_ID')
# zoom in into 'SALES' and 'QUANTITYORDERED', you will be able to see the monthly information color coded on the graph