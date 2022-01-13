from math import gamma
from os import sep
from urllib import request
from nltk import data
import pandas as pd
import string
import re
import numpy as np
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import preprocessing, model_selection, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import nltk
import pickle
nltk.download('prunkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time
from flask import jsonify, request
import json


from flask import Flask, render_template
from pandas.core.indexes import category
app = Flask(__name__)

classifierSVM= svm.SVC(C=0.5, degree=2, gamma='scale', kernel='sigmoid')
classifierNB = MultinomialNB(fit_prior=False)
cv = CountVectorizer()
tfidconverter = TfidfTransformer()


def labelToNumeric(category):
    if category == 'Positif':
        return 0
    elif category == 'Negatif':
        return 1
    else:
        return 0

# Index Page
@app.route("/")
def main():
    file = '/home/pythonku/flask-app/data/Dataset.csv'

    data = dataReading(file)
    positif, negatif = data['category'].value_counts()
    total=positif + negatif
    active_page="dashboard"
    
    return render_template('index.html',total=total,positif=positif,negatif=negatif,active_page=active_page)

# Preprocessing Page
@app.route('/preprocess', methods=['GET','POST'])
def preprocess():
    active_page = "preprocess"
    return render_template('preprocessing.html',active_page=active_page)

@app.route('/datapreprocess',methods=['GET','POST'])
def datapreprocess():
    file = '/home/pythonku/flask-app/data/text_preprocessing.csv'

    data = dataReading(file)
    result = data.to_json(orient="split")

    return result

#TF-IDF
@app.route('/pembobotan', methods=['GET','POST'])
def tfidf():

    file = '/home/pythonku/flask-app/data/data_after_preprocessing.csv'

    data= dataReading(file)
    positif, negatif = data['category'].value_counts()
    total = positif + negatif
    active_page='tfidf'
     
    return render_template('pembobotan.html', active_page=active_page)

# SVM
@app.route('/svm', methods=['GET','POST'])
def analisisSVM():
    active_page = 'svm'
    return render_template('svm.html',active_page=active_page)

@app.route('/klasifikasisvm', methods=['GET','POST'])
def klasifikasisvm():
    file = '/home/pythonku/flask-app/data/data_after_preprocessing.csv'

    data = dataReading(file)
    positif, negatif = data['category'].value_counts()
    total = positif + negatif

    X = cv.fit_transform(data['text_stemming'].values.astype('U')).toarray()
    Y = data['category']

    X_train, X_test, y_train, y_test = splitDatabase(X,Y)


    # import
    t0 = time.time()
    classifierSVM.fit(X_train, y_train)
    t1 = time.time()
    y_predSVM = classifierSVM.predict(X_test)
    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1

    cm_SVM = (confusion_matrix(y_test, y_predSVM))
    acc_SVM = (accuracy_score(y_test, y_predSVM))    
    pcc_SVM = (precision_score(y_test, y_predSVM))
    rec_SVM = (recall_score(y_test, y_predSVM))
    f1_SVM = (f1_score(y_test,y_predSVM))
    
    dt = {
        "total": total,
        "positif": positif,
        "negatif": negatif,
        "time_train": time_train,
        "time_predict": time_predict,
        "cm": cm_SVM.tolist(),
        "acc": acc_SVM,
        "pcc": pcc_SVM,
        "rec": rec_SVM,
        "f1": f1_SVM
    }
    result = jsonify(dt)
    return result

@app.route('/naive', methods=['GET','POST'])
def analisisNB():
    active_page = 'naive'
    return render_template('naive.html', active_page=active_page)

@app.route('/klasifikasinaive',methods=['GET','POST'])
def klasifikasinaive():
    file = '/home/pythonku/flask-app/data/data_after_preprocessing.csv'

    data = dataReading(file)
    positif, negatif = data['category'].value_counts()
    total = positif + negatif

    X = cv.fit_transform(data['text_stemming'].values.astype('U')).toarray()
    Y = data['category']

    X_train, X_test, y_train, y_test = splitDatabase(X,Y)


    # import
    t0 = time.time()
    classifierNB.fit(X_train, y_train)
    t1 = time.time()
    y_predNB = classifierNB.predict(X_test)
    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1

    cm_NB = (confusion_matrix(y_test, y_predNB))
    acc_NB = (accuracy_score(y_test, y_predNB))    
    pcc_NB = (precision_score(y_test, y_predNB))
    rec_NB = (recall_score(y_test, y_predNB))
    f1_NB = (f1_score(y_test,y_predNB))
    
    dt = {
        "total": total,
        "positif": positif,
        "negatif": negatif,
        "time_train": time_train,
        "time_predict": time_predict,
        "cm": cm_NB.tolist(),
        "acc": acc_NB,
        "pcc": pcc_NB,
        "rec": rec_NB,
        "f1": f1_NB
    }
    result = jsonify(dt)
    return result

@app.route('/test',methods=['GET','POST'])
def testModel():
    active_page = 'test'
    return render_template('test.html', active_page=active_page)

@app.route('/katatest',methods=['GET','POST'])
def katatest():

    classifierSVM= svm.SVC(C=0.5, degree=2, gamma='scale', kernel='sigmoid')
    classifierNB = MultinomialNB(fit_prior=False)
    cv = CountVectorizer()
    tfidfconverter = TfidfTransformer()
    # import
    classifierSVM = pickle.load(open('/home/pythonku/flask-app/data/classifierSVM.model','rb'))
    classifierNB = pickle.load(open('/home/pythonku/flask-app/data/classifierNB.model', 'rb'))
    cv = pickle.load(open('/home/pythonku/flask-app/data/cv.model', 'rb'))
    tfidfconverter = pickle.load(open('/home/pythonku/flask-app/data/tfidfconverter.model','rb'))

    text = request.get_data()
    text = str(text,'UTF-8')
    text_sebelum_preprocess = text
    app.logger.info(text)
    app.logger.info(type(text))
    # Prediksi SVM
    text = textCleaning(text)
    text = wordTokenizeWrapper(text)
    text = wordStopwords(text)
    text = wordNormalization(text)
    text = TreebankWordDetokenizer().detokenize(text)
    text = prosesStem(text)
    text_sesudah_preprocess = text
    text = cv.transform([text]).toarray()
    text = tfidfconverter.transform(text).toarray()
    textSVM = classifierSVM.predict(text)
    if textSVM[0] == 0:
        hasilSVM = 'Sentimen Negatif'
    else:
        hasilSVM = 'Sentimen Positif'
    textNB = classifierNB.predict(text)
    if textNB[0] == 0:
        hasilNB = 'Sentimen Negatif'
    else:
        hasilNB = 'Sentimen Positif'    
    
    # Save 
    pickle.dump(classifierSVM,open('/home/pythonku/flask-app/data/classifierSVM.model','wb'))
    pickle.dump(classifierNB,open('/home/pythonku/flask-app/data/classifierNB.model','wb'))
    pickle.dump(cv, open('/home/pythonku/flask-app/data/cv.model','wb'))
    pickle.dump(tfidfconverter, open('/home/pythonku/flask-app/data/tfidfconverter.model','wb'))
    
    dt = {
        'sebelum' : text_sebelum_preprocess,
        'sesudah' : text_sesudah_preprocess,
        'hasilSVM': hasilSVM,
        'hasilNB' : hasilNB
    }

    return jsonify(dt)



def dataReading(file):
    data = pd.read_csv(file, sep=';', encoding="latin-1")
    data['tweet'].str.encode('ascii','ignore')    
    data['category'] = data['Value'].apply(labelToNumeric)
    return data

def textCleaning(text):
    # Remove tab, New Line, and Back Line
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # Remove non ASCII (emoticon, chinese word, etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # Remove mention, link, hastag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # Remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    # Remove Number
    text = re.sub(r"\d+", "", text)
    # Remove Punctuation
    text = text.translate(str.maketrans("","",string.punctuation))
    # whitespace
    text = text.strip()
    # whitiespace multiple
    text = re.sub('\s+',' ',text)
    #single char
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    text = str.lower(text)
    return text

def wordTokenizeWrapper(text):
    return word_tokenize(text)

def wordStopwords(words):
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])
    txt_stopwords = pd.read_csv('/home/pythonku/flask-app/data/stopwords.txt',names=['stopwords'], header=None)
    list_stopwords.extend(txt_stopwords['stopwords'][0].split(' '))
    list_stopwords = set(list_stopwords)
    return [word for word in words if word not in list_stopwords]
    
def wordNormalization(document):
    normalization_word = pd.read_csv('/home/pythonku/flask-app/data/colloquial-indonesian-lexicon.csv')
    normalization_word_dict = {}

    for index, row in normalization_word.iterrows():
        if row[0] not in normalization_word_dict:
            normalization_word_dict[row[0]] = row[1]

    return [normalization_word_dict[term] if term in normalization_word_dict else term for term in document]

def pembobotan(data):
    X = cv.fit_transform(data['text_stemming'].values.astype('U')).toarray()
    Y = data['category']
    tfidConverter = TfidfTransformer()
    X = tfidConverter.fit_transform(X).toarray()
    return X, Y

def prosesStem(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem = stemmer.stem(text)
    return stem

def crossValidation(data, X, Y):
    clfSVM = svm.SVC(C=0.5, degree=2, gamma='scale', kernel='sigmoid')
    scoresSVM = cross_val_score(clfSVM, X, Y, cv=10)

    clfNaive = MultinomialNB(fit_prior=False)
    scoresNaive = cross_val_score(clfNaive, X, Y, cv=10)
    return scoresSVM, scoresNaive

def splitDatabase(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=45)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    app.run(debug=True)	
