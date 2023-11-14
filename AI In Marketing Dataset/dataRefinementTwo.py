import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

sales_df = pd.read_csv('AI In Marketing Dataset\sales_data_sample.csv', encoding = 'unicode_escape')

sales_df['COUNTRY'].value_counts().index

sales_df['COUNTRY'].value_counts()

# A function that visualizes the number of items in a given column.
# Please keep in mind that Plotly is a Python graphing toolkit that generates interactive, publication-quality graphs.

def barplot_visualization(x):
    fig = px.bar(
        x=sales_df[x].value_counts().index,
        y=sales_df[x].value_counts(),
        color=sales_df[x].value_counts().index,
        height=600
    )
    fig.show()

# Let us call this method for any specified column, such as 'COUNTRY'.

df = pd.DataFrame({
    'COUNTRY': ['US', 'CA', 'US', 'UK', 'FR']
})

# Call the barplot_visualization function
barplot_visualization('COUNTRY')

data = {'STATUS': ['SHIPPED', 'COMPLETED', 'CANCELLED', 'PENDING']}
df = pd.DataFrame(data)

# Calculate the number of unique status values
num_unique_status = df['STATUS'].nunique()

# Print the result
print("Number of unique order status values:", num_unique_status)

barplot_visualization('STATUS')

sales_df.drop(columns= ['STATUS'], inplace = True)
sales_df

barplot_visualization('PRODUCTLINE')

barplot_visualization('DEALSIZE')

# Function to add dummy variables to replace categorical variables

def dummies(x):
  dummy = pd.get_dummies(sales_df[x])
  sales_df.drop(columns = x , inplace = True)
  return pd.concat([sales_df, dummy], axis = 1)

# Let's obtain dummy variables for the column 'COUNTRY'
sales_df = dummies('COUNTRY')
sales_df

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
sales_df
