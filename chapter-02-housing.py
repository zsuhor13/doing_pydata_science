from io import StringIO
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")
print("pandas version {}".format(pd.__version__))

plt.style.use("ggplot")
#pd.options.display.mpl_style = "default"
# Read the file or the 'content'
#df = pd.read_csv(StringIO(resp.content))
print('---------------------------------------------------------------')
print('Initial state')
df = pd.read_excel("dds_datasets/dds_ch2_rollingsales/rollingsales_manhattan.xls", skiprows=4)
print(df.head())

#df.columns.str.replace(r'\s+|\\n', '_', regex=True)
df = df.rename(lambda x: x.replace('\n', '_'), axis='columns')
print('---------------------------------------------------------------')
print('Data cleaning vol 1')
print('Dropna and replace whitespaces')

print(df.head())

# Remove empty cells
df['NEIGHBORHOOD'] = df['NEIGHBORHOOD'].replace(r'^\s+$', 'Empty Neighborhood', regex=True)
df = df.replace(r'^\s*$', np.nan, regex=True)
#df = df.dropna()

# Only include the actual sales

df = df[df['SALE_PRICE'] > 0]

#df.groupby('NEIGHBORHOOD')['SALE_PRICE'].sum()

print(df.head(50))
#df.hist()
fig = df.plot.hist(column=['SALE_PRICE'], figsize=(10, 8))
#df['SALE_PRICE'].plot(kind='kde')

plt.show()