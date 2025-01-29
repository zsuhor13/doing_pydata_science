# %% 
import pandas as pd
#dataset fra kaggle https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present
#hele datasett 4.6GB, leser bare utvalg
df = pd.read_csv("data/nyt-articles-21m-2000-present/nyt-metadata.csv", nrows=999999)
# %%
df= df[['abstract','pub_date', 'section_name']]
# %%
df.section_name.value_counts()
# %%
sections_to_keep = [
    "Sports",
    "World",
    "U.S.",
    "Arts",
    "Books"
]
df = df[(df.section_name.isin(sections_to_keep)) & \
    ~(df.abstract.isna())]
# %%
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import re
nltk.download('stopwords')

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))
# Function to preprocess text
def preprocess_text(text):    
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text
df['abstract_features'] = df['abstract'].apply(preprocess_text)

## manuelt lagret etter prosessering her
#%% reload after preprocessing
df = pd.read_csv("data/nyt_processed.csv")
df = df.dropna()

# %%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(df["abstract_features"])

# %%

from sklearn.model_selection import train_test_split
y = df.pop('section_name')
X_train,X_test,y_train,y_test = train_test_split(X_counts,y,test_size=0.2)

# %%

from sklearn.naive_bayes import MultinomialNB
import numpy as np
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb_prediction = mnb.predict(X_test)


# %%
from sklearn.metrics import classification_report
print(classification_report(y_test, mnb_prediction))
# %%
def print_top10(vectorizer, clf: MultinomialNB, class_labels):
    """Prints features with the highest coefficient values, per class"""
    
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.feature_log_prob_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))
print_top10(vectorizer, mnb, mnb.classes_)
# %%
