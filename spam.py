
import numpy as np
import pandas as pd
data=pd.read_csv('Spam_doc.csv',encoding = "ISO-8859-1")
data['spam']=data['type'].map({'spam':1,'ham':0}).astype(int)
def tokenizer(text):
    return text.split()
data['text']=data['text'].apply(tokenizer)
from nltk.stem.snowball import SnowballStemmer
port_it=SnowballStemmer("english",ignore_stopwards=False)
def stem_it(text):
    return [port_it.stem(word) for word in text]
data['text']=data['text'].apply(stem_it)    
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmit_it(text):
    return [lemmatizer.lemmatize(word,pos="a") for word in text]
data['text']=data['text'].apply(lemmit_it)
from nltk.corpus import stopwords
stop_words=stopwords.words("english")
def stop_it(text):
    review=[word for word in text if not word in stop_words]
data['text']=data['text'].apply(stop_it)    
from sklearn.feature_extraction.text import TfidVectorizer
tfidf=TfidVectorizer()
y=data.spam.values
x=tfidf.fit_transform(data['text'])
from sklearn.model_selection import train_test_split
x_train,x_text,y_train,y_text=train_test_split(x,y,random_state=1,test_size=0.2,shuffle=False)
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_text)
from sklearn.metrics import accuracy_score
acc_linear_svc=accuracy_score(y_pred,y_text)*100
print("accuracy:",acc_linear_svc)