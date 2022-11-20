import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')

import numpy
import re
import sys
import os
import math

def tokenize(sentences):
  words = []
  w = word_extraction(sentences)
  words.extend(w)
  words = sorted(list(set(words)))
  return words

def word_extraction(sentence):
  ignore = list(stopwords.words('english'))
  words = re.sub("[^\w]", " ",  sentence).split()
  cleaned_text = [w.lower() for w in words if len(w)>2 and w not in ignore]
  return cleaned_text  

def generate_vocab(allsentences):
  allsentences=' '.join(allsentences)
  vocab = tokenize(allsentences)
  return vocab

def train(path): 
  classes = ['spam', 'ham']
  dictionary=[]
  textspam=[]
  textham=[]
  # N=0
  Nspam=0
  Nham=0
  Nctspam=0
  Nctham=0
  Ntspam=0
  Ntham=0
  cond_prob_spam=dict()
  cond_prob_ham=dict()
  dictionary_class=[]

  for i in classes:
    for j in os.listdir(path+i):
      if i=='spam':
          Nspam+=1
      else:
          Nham+=1

      file1 = open(path+i+"/"+j, errors="ignore")
      file=file1.read().split("\n")
      dictionary_class.append(i)
      dictionary.append(' '.join(file))
      # N+=1

  # print(dictionary)
  prior_spam = Nspam/(Nspam+Nham)
  prior_ham = Nham/(Nspam+Nham)

  V = generate_vocab(dictionary)
  for t in V:
    count=0
    
    for sent in dictionary:
      if t in sent:
        if dictionary_class[count]=='spam':
          Nctspam+=1
        else:
          Nctham+=1
      count+=1

    cond_prob_spam[t]=(Nctspam+1)/(Nspam+2)
    cond_prob_ham[t]=(Nctham+1)/(Nham+2)

    #print(cond_prob_spam, cond_prob_ham)
  return V, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham

for d in os.listdir("Dataset/"):
  path = "Dataset/"+d+"/train/"
  print("Training")
  V, prior_spam, prior_ham, cond_prob_spam, cond_prob_ham = train(path)
  path = "Dataset/"+d+"/train/"
  classes = ['spam', 'ham']
  actual_list=[]
  pred_list=[]

  for i in classes:
    for j in os.listdir(path+i):
      file1 = open(path+i+"/"+j, errors="ignore")
      file=file1.read().split("\n")

      W=generate_vocab(file)

      if i=='spam':
        actual_list.append(1)
      else:
        actual_list.append(0)
      
      score_spamm = math.log(prior_spam)
      score_hamm = math.log(prior_ham)

      for t_test in V:
        if t_test in W:
          if cond_prob_spam.get(t_test):
            score_spamm+=math.log(cond_prob_spam.get(t_test))
          if cond_prob_ham.get(t_test):
            score_hamm+=math.log(cond_prob_ham.get(t_test))

        else:
          if cond_prob_spam.get(t_test) and cond_prob_spam.get(t_test)<1:
            score_spamm+=math.log(1-cond_prob_spam.get(t_test))
          if cond_prob_ham.get(t_test) and cond_prob_ham.get(t_test)<1:
            score_hamm+=math.log(1-cond_prob_ham.get(t_test))

      #print(score_spamm, score_hamm)

      if score_spamm>=score_hamm:
        pred_list.append(1)
      else:
        pred_list.append(0)

  accuracy = len([actual_list[i] for i in range(0, len(actual_list)) if actual_list[i] == pred_list[i]]) / len(actual_list)
  print("Testing Accuracy on ", d, "is", accuracy)

  from sklearn.metrics import classification_report, confusion_matrix

  print("\nClassification Report\n", classification_report(actual_list, pred_list))


  print("\nConfusion Matrix\n", confusion_matrix(actual_list, pred_list))

