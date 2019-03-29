import re
from nutrition.structure.environment import PROJECT_FOLDER
import string
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

class NewsClassifier:

    @staticmethod
    def predict_cat(title):
        cat_names = {'b': 'business', 't': 'science and technology', 'e': 'entertainment', 'm': 'health'}
        cod = nb.predict(vectorizer.transform([title]))
        probability = nb.predict_proba(vectorizer.transform([title]))
        print("Printing probablity ... ")
        print(probability[0])
        # print("Printed probablity ... ")
        # print("Printing Inverse Transform ..... ")
        print(cod)
        # print(encoder.inverse_transform(cod))
        # print("Printed Inverse Transform ..... ")
        if probability[0][cod] < 0.90:
            return 'Other'
        else:
            return cat_names[encoder.inverse_transform(cod)[0]]

    @staticmethod
    def clean_text(s):
        s = s.lower()
        for ch in string.punctuation:
            s = s.replace(ch, " ")
        s = re.sub("[0-9]+", "||DIG||", s)
        s = re.sub(' +', ' ', s)
        return s

np.random.seed(123456)
dataset = pd.read_csv(PROJECT_FOLDER+"/test_data/uci-news-aggregator.csv")
dataset.head()

ob = NewsClassifier()
dataset['TEXT'] = [ob.clean_text(s) for s in dataset['TITLE']]

# pull the data into vectors
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dataset['TEXT'])

encoder = LabelEncoder()
y = encoder.fit_transform(dataset['CATEGORY'])

# split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

nb = MultinomialNB()
nb.fit(x_train, y_train)

results_nb_cv = cross_val_score(nb, x_train, y_train, cv=10)
print(results_nb_cv.mean())

nb.score(x_test, y_test)
print(nb.score(x_test, y_test))
x_test_pred = nb.predict(x_test)
print(x_test_pred)
confusion_matrix(y_test, x_test_pred)
print(confusion_matrix(y_test, x_test_pred))
print(classification_report(y_test, x_test_pred, target_names=encoder.classes_))
