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
df = pd.read_csv("nyt-data/nyt1.csv")
print(df.head())

df['Gender'] = df['Gender'].apply(lambda x: 'male' if x else 'female')
print('---------------------------------------------------------------')
print('Clean data')
print(df.head())

print('---------------------------------------------------------------')
print('Categorize age groups')
age_range = [0, 19, 25, 35, 45, 55, 65, np.inf]
age_labels = ['18--', '18-24', '25-34', '35-44', '45-54', '55-64', '65++']
pd.cut(df['Age'], bins=age_range, right=False, labels=age_labels)
df['Age_group'] = pd.cut(df['Age'], bins=age_range, right=False, labels=age_labels)
print(df.head())
df.hist()

print('---------------------------------------------------------------')
print('Click through rate by age groups')
#df.groupby('Age_group')[['Impressions', 'Clicks']].sum()

df.groupby('Age_group')['Age'].agg([len,  np.min, np.mean, np.max])
print(df.head(50))
print('---------------------------------------------------------------')

df.hist()