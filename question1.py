from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import csv
import os

# import statments
import numpy as np
import numpy
import re
import sys
import math

def tokenize(sentences):
  words = []
  for sentence in sentences:
    w = word_extraction(sentence)
    words.extend(w)
  # words = sorted(list(set(words)))
  return words

def word_extraction(sentence):
  ignore = ['a', "the", "is"]
  words = re.sub("[^\w]", " ",  sentence).split()
  cleaned_text = [w.lower() for w in words if len(w)>2 and w not in ignore]
  return cleaned_text  
  
def generate_bernoulli(allsentences):
  S=[]
  vocab = tokenize(allsentences)
  vocab = sorted(list(set(vocab)))
  print(len(vocab))

  for sentence in allsentences:
    words = word_extraction(sentence)
    bag_vector = numpy.zeros(len(vocab))

    for w in words:
        for i,word in enumerate(vocab):
            if word == w: 
                bag_vector[i] = 1
                      
    S.append(numpy.array(bag_vector))
    numpy.set_printoptions(threshold=sys.maxsize)
  return S, vocab

def generate_bow(allsentences):
  S=[]
  vocab = tokenize(allsentences)
  vocab = sorted(list(set(vocab)))

  for sentence in allsentences:
    words = word_extraction(sentence)
    bag_vector = numpy.zeros(len(vocab))

    for w in words:
        for i,word in enumerate(vocab):
            if word == w: 
                bag_vector[i] += 1
                      
    S.append(numpy.array(bag_vector))
    numpy.set_printoptions(threshold=sys.maxsize)
  return S, vocab

#dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def train():
  for d in os.listdir("Dataset/"):
    print(d)
    path = "Dataset/"+d+"/train/"
    dictionary=[]
    classes=['spam', 'ham']
    lbl=[]
    for i in classes:
      for j in os.listdir(path+i):

        file1 = open(path+i+"/"+j, errors="ignore")
        file=file1.read().split("\n")
        dictionary.append(' '.join(file))
        if i=='spam':
          lbl.append(1)
        else:
          lbl.append(0)

    X_bow, vocab_bow=generate_bernoulli(dictionary)
    X_bern, vocab_bern=generate_bow(dictionary)
    print("Bow", X_bow)
    print("Bern", X_bern)

train()