
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline


df = pd.read_csv("new_data.csv")

df.head()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], random_state=0)

model = Pipeline([("vect", CountVectorizer()),("clf",LogisticRegression())])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('AUC: ', roc_auc_score(y_test, predictions))

filename = 'finalized_model.sav'
joblib.dump(model, filename)