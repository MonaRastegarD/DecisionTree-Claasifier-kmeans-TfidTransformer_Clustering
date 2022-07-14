import pandas as pd

# Read the input csv file
df = pd.read_excel('dataset.xlsx', index_col = 0)
annual_averages = df.groupby('feature 0')['feature 1', 'feature 2','feature 3'].mean()

annual_averages.plot()

f0 = df['feature 0'].array
f2 = df['feature 2'].array
f3 = df['feature 3'].array
f4 = df['feature 4'].array
f5 = df['feature 5'].array
f6 = df['feature 6'].array
f7 = df['feature 7'].array
f8 = df['feature 8'].array
f9 = df['feature 9'].array
f10 = df['feature 10'].array
f11 = df['feature 11'].array
f12 = df['feature 12'].array
f13 = df['feature 13'].array
f14 = df['feature 14'].array
f15 = df['feature 15'].array
f16 = df['feature 16'].array
f17 = df['feature 17'].array
f18 = df['feature 18'].array
f19 = df['feature 19'].array



import pandas as pd
import numpy as np


data=pd.read_excel('dataset.xlsx') 
idea=data.iloc[:,0:1] #Selecting the first column that has text. 

#Converting the column of data from excel sheet into a list of documents, where each document corresponds to a group of sentences.
corpus=[]
for index,row in idea.iterrows():
    corpus.append(row['_index_text_data']) 

#Count Vectoriser then tidf transformer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus) #ERROR AFTER EXECUTING THESE #LINES

#vectorizer.get_feature_names()

#print(X.toarray())     

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape )                        

from sklearn.cluster import KMeans

num_clusters = 5 #Change it according to your data.
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()

idea={'Idea':corpus, 'Cluster':clusters} #Creating dict having doc with the corresponding cluster number.
frame=pd.DataFrame(idea,index=[clusters], columns=['Idea','Cluster']) # Converting it into a dataframe.

print("\n")
print(frame) #Print the doc with the labeled cluster number.
print("\n")
print(frame['Cluster'].value_counts()) #Print the counts of doc belonging `#to each cluster.`


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Read the input csv file
dataset = pd.read_csv("dataset.xlsx")

# Drop the  since this is not a good feature to split the data on
dataset = dataset.drop("feature 19", axis=1)

# Split the data into features and target
features = dataset.drop("feature 11", axis=1)
targets = dataset["feature 11"]

# Split the data into a training and a testing set
train_features, test_features, train_targets, test_targets = \
        train_test_split(features, targets, train_size=0.75)


from pyxll import xl_func
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import os

@xl_func("float, int, int: object")
def ml_get_zoo_tree(train_size=0.75, max_depth=5, random_state=245245):
    # Load the zoo data
    dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.xlsx"))

    # Drop the names since this is not a good feature to split the data on
    dataset = dataset.drop("feature 16", axis=1)

    # Split the data into a training and a testing set
    features = dataset.drop("feature 7", axis=1)
    targets = dataset["feature 9"]

    train_features, test_features, train_targets, test_targets = \
        train_test_split(features, targets, train_size=train_size, random_state=random_state)

    # Train the model
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    tree = tree.fit(train_features, train_targets)

    # Add the feature names to the tree for use in predict function
    tree._feature_names = features.columns

    return tree

from pyxll import xl_func, async_call

@xl_func("object tree, dict features: var")
def ml_zoo_predict(tree, features):
    # Convert the features dictionary into a DataFrame with a single row
    features = pd.DataFrame([features], columns=tree._feature_names)

    # Get the prediction from the model
    prediction = tree.predict(features)[0]

