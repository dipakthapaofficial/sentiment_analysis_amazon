from filecmp import BUFSIZE
import json
from multiprocessing import Pool
import os
import platform
import pandas as pd

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
from gensim.models import Word2Vec #Word2Vec is mostly used for huge datasets

from file_parser import getCsvFiles, getFiles

data_pool = pd.DataFrame()

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

#1. STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)


#2. STEMMING
 
# Initialize the stemmer
snow = SnowballStemmer('english')
def stemming(string):
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)

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
def lemmatizer(string, cores=6):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    
    # a = []
    
    # with Pool(processes=cores) as pool:
    #     # Initialize the lemmatizer
    #     wl = WordNetLemmatizer()
    #     word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    #     a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token

    return " ".join(a)



#FINAL PREPROCESSING
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
    return stopword(preprocess(string)) 

def classifyData(fileName):
    df_train_dataset = pd.read_json(fileName, lines = True)
    # df_train_dataset = pd.read_csv(fileName)

    print(df_train_dataset.shape)

    #remove rows with empty reviews
    df_train_dataset = df_train_dataset[df_train_dataset['reviewText'].notna()]
    df_train_dataset = df_train_dataset[df_train_dataset['reviewText'] != ""]
    df_train_dataset['overall'] = df_train_dataset['overall'].astype(object) # fix datatype error
    
    df_train_dataset['reviewText'] = df_train_dataset['reviewText'].astype(object) # fix datatype error
    
    #Add only useful data from dataset
    # dataset = {"reviewText": df_train_dataset["reviewText"], "overall": df_train_dataset["overall"]  }
    
    # df_train_dataset = pd.DataFrame(data = dataset)
    df_train_dataset = df_train_dataset.dropna()

    #Get shape of the data frame
    print(df_train_dataset.shape)

    #Print data frame from top
    print(df_train_dataset.head())
    

    #Split into multiple files
    # for id, df_i in  enumerate(np.array_split(df_train_dataset, 50)):
    #     actualFileName = fileName.split(".json")[0]
    #     df_i.to_csv(actualFileName + '_{id}.csv'.format(id=id))
    
    # return
    
    #df_train_dataset.to_csv(fileName, index=False)
    print("\n\n\n\n\n")

    #Filter out neutral i.e. rating 3 from the list to segregate between positive and negative reviews
    df_train_dataset["score"] = df_train_dataset["overall"].apply(lambda rating : +1 if str(rating) > '3' else -1)
    # data_pool = df_train_dataset
    print(df_train_dataset.head())

    return
    

    df_train_dataset['clean_text'] = df_train_dataset['reviewText'].apply(lambda x: finalpreprocess(x))
    # data_pool.head()

    
    # data_pool = data_pool.append(df_train_dataset, ignore_index = True)

    return df_train_dataset


if __name__ == "__main__":

    text = """
        The sun sets in the west signalling the end of the dusk to welcome the dark night bringing out mysteries engulfed in it.
    """

    finalText = lemmatizer(text)
    stemText = stemming(text)
    print(finalText)
    print("\n")
    print(stemText)
    tokenized_word = word_tokenize(text)
    print(tokenized_word)

    word_pos_tags = nltk.pos_tag(tokenized_word)

    print(word_pos_tags)
    
    # fileNames = getFiles("data")
    # fileNames = getCsvFiles("data/csv")

    # for fileName in fileNames:
    #     data_pool_classified = classifyData(fileName)

        # x = data_pool_classified['score'].value_counts()
        # print(fileName)
        # print(x)

        # actualFileName = fileName.split(".json")[0]
        # actualFileName = fileName.split(".csv")[0]

        # data_pool_classified.to_csv(actualFileName+"_final"+".csv", header=["reviewText", "overall", "score", "clean_text"], index = False, line_terminator='\n')



        # data_pool = data_pool.append(data_pool_classified, ignore_index = True)


    
    # data_pool_positive = data_pool[data_pool['score'] == 1]
    # print("positive=>",data_pool_positive.shape)
    # data_pool_positive.to_csv('positive.csv', header=["reviewText", "overall", "score"], index = False, line_terminator='\n')

    # data_pool_negative = data_pool[data_pool['score'] == -1]
    # data_pool_negative.to_csv('negative.csv', header=["reviewText", "overall", "score"], index = False, line_terminator='\n')

    # print("negative=>",data_pool_negative.shape)

    # data_pool.to_csv('cleaned_data_pool.csv', header=["reviewText", "overall", "score"], index = False, line_terminator='\n')
    # print("data_pool=>",data_pool.shape)

    # test_pool =  pd.read_csv('cleaned_data_pool.csv')
    # print(test_pool.shape)
    # print(test_pool)


