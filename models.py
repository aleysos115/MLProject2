import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import string
from textblob import TextBlob
from textblob import Word
from sklearn import datasets
from pathlib import Path

def preproccess_data(data):
    stop = stopwords.words('english')
    data[6] = data[6].apply(lambda x: " ".join(x.lower() for x in x.split()))
    print("Lower case: Done")
    data[6] = data[6].str.replace('[^\w\s]','')
    print("Removal of punctuation: Done")
    data[6] = data[6].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    print("Removal of Stop words: Done")
    data[6] = data[6].str.replace("urlLink", "")
    print("Removal of urlLink: Done")
    data[6] = data[6].apply(lambda x: str(TextBlob(x).correct()))
    print("Spelling Check: Done")
    data[6] = data[6].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    print("Lemmatize: Done")

    to16 = {"14": "13-16", "15": "13-16", "16": "13-16"}
    to26 = {"24": "24-26", "25": "24-26", "26": "24-26"}
    to36 = {"34": "34-36", "35": "34-36", "36": "34-36"}
    to46 = {"44": "44-46", "45": "44-46", "46": "44-46"}

    data[2] = multiple_replace(to16, data[2])
    data[2] = multiple_replace(to26, data[2])
    data[2] = multiple_replace(to36, data[2])
    data[2] = multiple_replace(to46, data[2])
    print("Age Correct: Done")

def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

def main():
    clean_data = Path('./data/train_clean.csv')
    if not clean_data.is_file():
        data = pd.read_csv('./data/train_raw.csv', header=None)
        preproccess_data(data)
        data.to_csv('./data/train_clean.csv')
    clean_train = pd.read_csv('./data/train_clean.csv', header=None)
    print(clean_train)


if __name__ == "__main__":
    main()
