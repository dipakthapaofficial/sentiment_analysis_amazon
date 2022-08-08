from distutils import extension
from filecmp import BUFSIZE
import io
import json
import os
import platform
import pandas as pd

data_pool = pd.DataFrame()


def getCsvFiles(location):
    files = os.listdir(location)
    filesToBeRead = []

    #Ignore all files except json files
    for reviewFile in files:
        fileExtention = reviewFile.split(".")
        if fileExtention[-1] == 'csv':
            if platform.system() == 'Windows':
                filesToBeRead.append(location +"\\" + reviewFile)
            else:
                filesToBeRead.append(location +"/" + reviewFile)
    
    print(filesToBeRead)

    return filesToBeRead

def getFiles(location):
    files = os.listdir(location)
    filesToBeRead = []

    #Ignore all files except json files
    for reviewFile in files:
        fileExtention = reviewFile.split(".")
        if fileExtention[-1] == 'json':
            if platform.system() == 'Windows':
                filesToBeRead.append(location +"\\" + reviewFile)
            else:
                filesToBeRead.append(location +"/" + reviewFile)
    
    print(filesToBeRead)

    return filesToBeRead
    

def classifyData(fileName):
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
    # data_pool = df_train_dataset

    # data_pool['clean_text'] = data_pool['reviewText'].apply(lambda x: finalpreprocess(x))
    # data_pool.head()

    
    # data_pool = data_pool.append(df_train_dataset, ignore_index = True)

    return df_train_dataset


if __name__ == "__main__":
    
    fileNames = getFiles("data")

    for fileName in fileNames:
        data_pool_classified = classifyData(fileName)

        x = data_pool_classified['score'].value_counts()
        print(fileName)
        print(x)

        data_pool = data_pool.append(data_pool_classified, ignore_index = True)

    
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


