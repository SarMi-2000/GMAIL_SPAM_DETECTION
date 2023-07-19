# GMAIL_SPAM_DETECTION
# Spam Classification Model

This repository contains code for a spam classification model that uses a logistic regression classifier and various text processing techniques to classify emails as either spam or ham (non-spam).

## Dataset

The dataset used for training and testing the model is stored in the `Spam_doc.csv` file. It contains email messages labeled as spam or ham. The `type` column contains the labels, which are mapped to binary values (1 for spam, 0 for ham) in the preprocessing stage.

## Preprocessing

The following preprocessing steps are performed on the text data:

- Tokenization: The email messages are split into individual words.
- Stemming: SnowballStemmer is used to reduce words to their base or root form.
- Lemmatization: WordNetLemmatizer is used to convert words to their base or dictionary form.
- Stopword Removal: Common English stopwords are removed from the text.

## Model Training

The processed text data is then converted into TF-IDF (Term Frequency-Inverse Document Frequency) vectors using the TfidfVectorizer. The dataset is split into training and testing sets, with 80% for training and 20% for testing.

A logistic regression classifier is trained on the training set using the sklearn library. The trained model is then used to predict the labels for the test set.

## Evaluation

The accuracy of the spam classification model is evaluated using the accuracy_score metric from sklearn.metrics. The accuracy percentage is printed to the console.

## Usage

To run the spam classification model, follow these steps:

1. Install the necessary dependencies listed in the `requirements.txt` file.
2. Ensure that the `Spam_doc.csv` dataset file is located in the same directory.
3. Run the `spam_classification.py` script.

## Dependencies

The project requires the following dependencies:

- numpy
- pandas
- nltk
- scikit-learn
