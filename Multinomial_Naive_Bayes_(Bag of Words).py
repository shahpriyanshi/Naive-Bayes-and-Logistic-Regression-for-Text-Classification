# -*- coding: utf-8 -*-
"""Final_MNB_BOW.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/142zaMmgcWEfxglXZnq-sdlwdyPCIcy74
"""

import numpy
import re
import sys
import os
import math
import nltk
nltk.download('punkt')

def bow(dictionary):
  sentences = nltk.sent_tokenize(''.join(dictionary))  
  words = nltk.word_tokenize(''.join(dictionary))  
  words=[w.lower() for w in words if w.isalpha() and len(w)>2]

  word_dict=dict()

  words = sorted(list(set(words)))
  for word in words:
      word_dict[word] = 1

  return words, word_dict

def freq(class_docs, vocab_dict):

  class_vocab_dict=vocab_dict
  #print("cd", class_vocab_dict)
  sentences = nltk.sent_tokenize(''.join(class_docs))  
  words = nltk.word_tokenize(''.join(class_docs))  
  words=[w.lower() for w in words if w.isalpha() and len(w)>2]
  total = len(words)

  for word in words:
    class_vocab_dict[word] = class_vocab_dict.get(word)+1

  #print("cd",class_vocab_dict)
  return class_vocab_dict, total


def train(path):
#     path = "Dataset/hw1/train/"
    classes = ['spam', 'ham']

    dictionary=[]
    textspam=[]
    textham=[]
    N=0
    Nspam=0
    Nham=0
    cond_prob_spam=dict()
    cond_prob_ham=dict()
    
    for i in classes:
        for j in os.listdir(path+i):
            if i=='spam':
                Nspam+=1
            else:
                Nham+=1

            file1 = open(path+i+"/"+j, errors="ignore")
            file=file1.read().split("\n")
            dictionary.append(' '.join(file))
            if i=='spam':
                textspam.append(' '.join(file))
            else:
                textham.append(' '.join(file))
            N+=1

    prior_spam = Nspam/N
    prior_ham = Nham/N

    V, vocab_dict = bow(dictionary)

    tctspam, t_spam = freq(textspam, vocab_dict)
    tct_spam=0
    
    for t in V:
        if tctspam.get(t):
            tct_spam += tctspam.get(t)
        else:
            pass
        
    for t in V:
        if tctspam.get(t):
            cond_prob_spam[t]=tctspam.get(t)/(tct_spam+t_spam)
    
    V, vocab_dict = bow(dictionary)
    tctham, t_ham = freq(textham, vocab_dict)
  
    tct_ham=0
    for t in V:
        if tctham.get(t):
            tct_ham += tctham.get(t)
        else:
            pass

    for t in V:
        if tctham.get(t):
            cond_prob_ham[t]=tctham.get(t)/(tct_ham+t_ham)

    #print(prior_spam, prior_ham, cond_prob_spam, cond_prob_ham)
    return V, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham

#Training

for d in os.listdir("Dataset/"):
    path = "Dataset/"+d+"/train/"
    V, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham = train(path)

    #testing
    path = "Dataset/"+d+"/train/"
    classes = ['spam', 'ham']
    actual_list=[]
    pred_list=[]

    for i in classes:
        for j in os.listdir(path+i):
            file1 = open(path+i+"/"+j, errors="ignore")
            file=file1.read().split("\n")
            W, _=bow(file)
            
            score_spamm = numpy.log(prior_spam)
            score_hamm = numpy.log(prior_ham)
            # print(score_spamm, score_hamm)

            if i=='spam':
                actual_list.append(1)
            
            else:
                actual_list.append(0)
                
            for t in W:
                f_temp=cond_prob_spam.get(t)
                f_temp1=cond_prob_ham.get(t)
                
                if f_temp:
                    # print(numpy.log(f_temp))
                    score_spamm+=numpy.log(f_temp)
                if f_temp1:
                    # print(numpy.log(f_temp1))
                    score_hamm+=numpy.log(f_temp1)

            # print(score_spamm, score_hamm)
            if score_spamm>=score_hamm:
                # print("Hi", score_spamm, score_hamm)
                pred_list.append(1)
            else:
                # print("OK", score_spamm, score_hamm)
                pred_list.append(0)

    accuracy = len([actual_list[i] for i in range(0, len(actual_list)) if actual_list[i] == pred_list[i]]) / len(actual_list)
    print("Testing accuracy on ", d,"is", accuracy)

    from sklearn.metrics import classification_report, confusion_matrix

    print("\nClassification Report\n", classification_report(actual_list, pred_list))
    print("\nConfusion Matrix\n", confusion_matrix(actual_list, pred_list))
