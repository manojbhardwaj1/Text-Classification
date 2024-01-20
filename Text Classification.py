import numpy as np
import pandas as pd

temp_df = pd.read_csv("D:\Data\imbd_dataset.csv")

df = temp_df.iloc[:10000]
df.head()
df['review'][1]
df['sentiment'].value_counts()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# Basic Preprocessing
# Remove tags
# lowercase
# remove stopwords

import re
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text

df['review'] = df['review'].apply(remove_tags)
df
df['review'] = df['review'].apply(lambda x:x.lower())

from nltk.corpus import stopwords
sw_list = stopwords.words('english')
df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))

df
X = df.iloc[:,0:1]
y = df['sentiment']
X
y

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train.shape

# Applying BoW
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()
X_train_bow.shape

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_bow,y_train)

y_pred = gnb.predict(X_test_bow)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test,y_pred)

cv = CountVectorizer(max_features=3000)
X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

rf = RandomForestClassifier()
rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test,y_pred)


cv = CountVectorizer(ngram_range=(1,2),max_features=5000)
X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

rf = RandomForestClassifier()
rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test,y_pred)


# Using TfIdf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train['review']).toarray()
X_test_tfidf = tfidf.transform(X_test['review'])

rf = RandomForestClassifier()
rf.fit(X_train_tfidf,y_train)
y_pred = rf.predict(X_test_tfidf)
accuracy_score(y_test,y_pred)


# Train the data 
import gensim

from nltk import sent_tokenize
from gensim.utils import simple_preprocess

story = []
for doc in df['review']:
    raw_sent = sent_tokenize(doc)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))

model = gensim.models.Word2Vec(
    window=10,
    min_count=2
)

model.build_vocab(story)
model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
len(model.wv.index_to_key)

def document_vector(doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc.split() if word in model.wv.index_to_key]
    return np.mean(model.wv[doc], axis=0)

document_vector(df['review'].values[0])

from tqdm import tqdm
X = []
for doc in tqdm(df['review'].values):
    X.append(document_vector(doc))

X = np.array(X)
X[0]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred) 
