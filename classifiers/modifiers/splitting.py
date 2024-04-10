from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# loading data
data = pd.read_csv (r'./classifiers/modifiers/data/processed_data.csv')

# splitting data into test and train data
train, test = train_test_split(data, test_size=0.1, random_state=42)

# categorization to identify and test
train_x = [x for x in train['phrase']] 
train_y = [x for x in train['type']]
test_x = [x.replace('[', '').replace(']', '') for x in test['phrase']]
test_y = [x for x in test['type']]

vectorizer = TfidfVectorizer()

# fit transforming train_x
vect_train_x = vectorizer.fit_transform(train_x)
vect_test_x = vectorizer.transform(test_x)

