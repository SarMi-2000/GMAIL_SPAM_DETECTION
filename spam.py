import numpy as np
import pandas as pd

# Read the data from the CSV file
data=pd.read_csv('SPAM.csv',encoding = "ISO-8859-1")

# Map 'spam' and 'ham' labels to binary values
data['spam'] = data['type'].map({'spam': 1, 'ham': 0}).astype(int)

# Tokenize the text
def tokenizer(text):
    return text.split()

data['text'] = data['text'].apply(tokenizer)

# Perform stemming using SnowballStemmer
from nltk.stem.snowball import SnowballStemmer
port_it = SnowballStemmer("english", ignore_stopwords=False)

def stem_it(text):
    return [port_it.stem(word) for word in text]

data['text'] = data['text'].apply(stem_it)

# Perform lemmatization using WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_it(text):
    return [lemmatizer.lemmatize(word, pos="a") for word in text]

data['text'] = data['text'].apply(lemmatize_it)

# Remove stopwords using NLTK's stopwords corpus
from nltk.corpus import stopwords
stop_words = stopwords.words("english")

def remove_stopwords(text):
    return [word for word in text if not word in stop_words]

data['text'] = data['text'].apply(remove_stopwords)

# Perform TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
y = data.spam.values
x = tfidf.fit_transform(' '.join(text) for text in data['text'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2, shuffle=False)

# Train a logistic regression classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Calculate the accuracy of the classifier
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test) * 100
print("Accuracy:", accuracy)
