import pandas as pd

## In this Example we will look int assorting the data and extraxting information. 

sales_df = pd.read_csv('AI In Marketing Dataset\sales_data_sample.csv', encoding = 'unicode_escape')

#MSRP is an abbreviation for Manufacturer's Suggested Retail Price, which is the suggested retail price of a product.
#The MSRP is used to set a consistent pricing for items across several business retail locations.

sales_df

# Consider the many sorts of data.
sales_df.dtypes

# Create a sample DataFrame with an 'ORDERDATE' column
data = {'ORDERDATE': ['2020-01-01', '2021-01-02', '2022-01-11']}
df = pd.DataFrame(data)

# Convert 'ORDERDATE' column to datetime format
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])

# Verify the conversion by checking the datatype
print(df['ORDERDATE'].dtype)

data = {'ADDRESSLINE2': ['123 Main St', None, '456 Elm St']}
df = pd.DataFrame(data)

# Check the number of null values in the 'ADDRESSLINE2' column
null_values = df['ADDRESSLINE2'].isnull().sum()

# Print the number of null values
print(null_values)

# Check the number of non-null values in the dataframe
sales_df.info()

# Check the number of Null values in the data
sales_df.isnull().sum()

# since there are lot of Null values in 'addressline2', 'state', 'postal code' and 'territory' we can drop them.
# Country would represent the order grographical information.
# Also we can drop city, address1, phone number, contact_name, contact last_name and contact first_name since they are not required for the analysis


"""MINI CHALLENGE #3:
- How many unique values exist in 'country'?
- How many unique product codes and product lines do we have?
"""

# Obtain the number of unique values in each column
sales_df.nunique()

# Create a sample DataFrame with 'country', 'product_code', and 'product_line' columns
data = {'country': ['US', 'CA', 'US', 'UK', 'FR'],
        'product_code': ['A123', 'B456', 'C789', 'D1011', 'E1213'],
        'product_line': ['Electronics', 'Home Goods', 'Apparel', 'Toys', 'Sports']}
df = pd.DataFrame(data)

# Count the number of unique values in 'country'
num_unique_countries = df['country'].nunique()

# Count the number of unique product codes
num_unique_product_codes = df['product_code'].nunique()

# Count the number of unique product lines
num_unique_product_lines = df['product_line'].nunique()

# Print the results
print("Number of unique countries:", num_unique_countries)
print("Number of unique product codes:", num_unique_product_codes)
print("Number of unique product lines:", num_unique_product_lines)