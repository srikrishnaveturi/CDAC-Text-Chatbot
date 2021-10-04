# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:09:59 2021

@author: vetur
"""

import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
with open("intents.json") as file:
    data = json.load(file)
    
try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokenizedWords = nltk.word_tokenize(pattern)
            words.extend(tokenizedWords)
            docs_x.append(tokenizedWords)
            docs_y.append(intent["tag"])
            
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    training = []
    output = []
    
    out_empty = [0 for _ in range(len(labels))]
    
    for x,doc in enumerate(docs_x):
        bag = []
        stemmedWords = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in stemmedWords:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output),f)
import keras
from keras.models import Sequential
from keras.layers import Dense

try:
    classifier = keras.models.load_model("chatbotModel.h5")
except:     
    #initializing the ANN
    classifier = Sequential()
    
    #adding the  input layer and the first hidden layer
    classifier.add(Dense(units=8, input_dim = len(training[0]), activation='relu', kernel_initializer="uniform"))
    
    #adding the second hidden layer
    classifier.add(Dense(units=8, activation='relu', kernel_initializer="uniform"))
    classifier.add(Dense(units=len(output[0]), activation="softmax", kernel_initializer="uniform"))
    
    #compiling the ANN(training)
    
    classifier.compile(optimizer = "adam", loss = "categorical_crossentropy",metrics=["accuracy"])
    #the loss function is the value that will be optimised(mostly minimised) by the optimiser
    
    #fitting the classifier with X_train and y_train
    classifier.fit(training,output,batch_size = 16,epochs = 300)
    
    classifier.save("chatbotModel.h5")

    
def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i] = 1
    bag = numpy.array(bag)
    bag = numpy.resize(bag,(1,len(training[0])))
    return bag
    
def chat():
    print("The bot is listening, type 'quit' to stop")
    while True:
        inp = input("You : ")
        if inp == "quit":
            break
        
        results = classifier.predict(bagOfWords(inp, words))
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in data['intents']:
            if tg["tag"] == tag:
                responses = tg["responses"]
                
        print(random.choice(responses))
        
        

chat()

