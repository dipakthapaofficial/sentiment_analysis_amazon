import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import re

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.corpus import stopwords

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec
import time

from file_parser import getFiles

data_pool = pd.DataFrame()

def computeScore(data):
    if data < 3:
        return -1
    elif data > 3:
        return 1
    elif data == 3:
        return 0
    else:
        return 10

def preprocess(text):
    
    text = text.lower()#lowercase text
    text=text.strip() #get rid of leading/trailing whitespace 
    text=re.compile('<.*?>').sub('', text) #Remove HTML tags/markups
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  #Replace punctuation with space. Careful since punctuation can sometime be useful
    text = re.sub('\s+', ' ', text)   #Remove extra space and tabs
    text = re.sub(r'\[[0-9]*\]',' ',text)   #[0-9] matches any digit (0 to 10000...)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip()) 
    text = re.sub(r'\d',' ',text)  #matches any digit from 0 to 100000..., \D matches non-digits
    text = re.sub(r'\s+',' ',text)   #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    return text

def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#3. LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
# Full list is available here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def cleanText(text):
    # print("\n")
    # print(text)
    if not isinstance(text, str):
        print(text)

    if '"' in text:
        text.replace('"','')
    if '\n' in text:
        text.replace('\n', '')

    if "\\" in text:
        text.replace('\\', '')
    
    return text

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
#     return removeStopword(preprocess(string))


def classifier(fileName, data_pool, test_data_pool=None):
    df_train_dataset = pd.read_json(fileName, lines = True)

    print(df_train_dataset.shape)

    #remove rows with empty reviews
    df_train_dataset = df_train_dataset[df_train_dataset['reviewText'].notna()]
    df_train_dataset = df_train_dataset[df_train_dataset['reviewText'] != ""]
    df_train_dataset['overall'] = df_train_dataset['overall'].astype(object) # fix datatype error
    
    df_train_dataset['reviewText'] = df_train_dataset['reviewText'].astype(object) # fix datatype error
    
    #Add only useful data from dataset
    dataset = {"reviewText": df_train_dataset["reviewText"], "overall": df_train_dataset["overall"]  }
    
    df_train_dataset = pd.DataFrame(data = dataset)
    df_train_dataset = df_train_dataset.dropna()

    #Get shape of the data frame
    print(df_train_dataset.shape)

    #Print data frame from top
    print(df_train_dataset.head())
    
    #df_train_dataset.to_csv(fileName, index=False)

    #Filter out neutral i.e. rating 3 from the list to segregate between positive and negative reviews
    df_train_dataset["score"] = df_train_dataset["overall"].apply(lambda rating : +1 if str(rating) > '3' else -1)
    data_pool = df_train_dataset

    data_pool['clean_text'] = data_pool['reviewText'].apply(lambda x: finalpreprocess(x))
    data_pool.head()

    
    data_pool = data_pool.append(df_train_dataset, ignore_index = True)

    #define vectorizer
    vectorizer = TfidfVectorizer()
        
    x_train = vectorizer.fit_transform(data_pool["reviewText"])

    # print(x_train)
    # print(vectorizer.vocabulary_)


    #Train y element
    y_train = data_pool["score"]
    # print(y_train)

    start_time = time.time() 

    logistic_regression_model = LogisticRegression(solver='lbfgs', max_iter=100)
    logistic_regression_model.fit(x_train, y_train)



    #Test model
    x_test = vectorizer.transform(test_data_pool["reviewText"])
    y_test = test_data_pool["score"]

    logistic_regression_model.score(x_test, y_test)

    # testcases = [
    # "this is quite bad and disappointing",
    # "quite happy with my purchase",
    # "neutral"
    # ]
    # x_testcases = vectorizer.transform(testcases)
    # print(logistic_regression_model.predict(x_testcases))



    #FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
    #It's a probabilistic classifier that makes use of Bayes' Theorem, a rule that uses probability to make predictions based on prior knowledge of conditions that might be related. This algorithm is the most suitable for such large dataset as it considers each feature independently, calculates the probability of each category, and then predicts the category with the highest probability.
    

    end_time = time.time()
    print("Total time to train and predict using Naive Bayes=",(start_time-end_time))



# if __name__ == "__main__":
fileNames = getFiles("data")

for fileName in fileNames:
    test_data_pool = pd.read_json("./data/sample.json1", lines = True)

    print(test_data_pool.shape)

    test_data_pool = test_data_pool[test_data_pool['reviewText'].notna()]
    test_data_pool = test_data_pool[test_data_pool['reviewText'] != ""]

    print(test_data_pool.shape)


    test_data_pool['score'] = test_data_pool["overall"].apply(lambda x: computeScore(x))
    test_data_pool['reviewText'] = test_data_pool["reviewText"].apply(lambda x: cleanText(x))


    classifier(fileName, data_pool, test_data_pool)




    
