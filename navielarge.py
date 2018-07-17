#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:59:59 2018

Sentimental analysis using navie base classification technique

@author: harshabm
"""

import csv
from re import sub
import numpy as np
from sklearn.cross_validation import train_test_split

dataset = list()
fp=open('/home/harshabm/Documents/Placements/Projects/Sentimental analysis/Dataset/yelp_labelled.txt','r')
reader = csv.reader(fp , delimiter='\t')

for row in reader:
    dataset.append(row)
    
x = list() 
y = list()
   
for i in range(len(dataset)):
    x.append(dataset[i][0])
    y.append(dataset[i][1])
    
y = np.array(y,dtype = int)

x_modified = list()

for i in x:
    temp = sub('[^A-Za-z" "]+', '', str(i))
    temp = temp.lower()
    x_modified.append(temp)

X_train, X_test, Y_train, Y_test = train_test_split(x_modified,y,test_size=0.25,random_state=2)
Y_train = Y_train.reshape(1,len(Y_train))
Y_test = Y_test.reshape(1,len(Y_test))

data = list()
for i in range(len(X_train)):
    temp = X_train[i].split(' ')
    data.append(temp)

unique_words = list()
for i in data:
    for j in i:
        if j not in unique_words:
            unique_words.append(j)
        
stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

for i in unique_words:
    if(str(i) in stopwords):
        unique_words.remove(i)


#generating the frequency tabel ie how many times did the word appears in each class.
freqtable = list()
ctr1 = 0
ctr2 = 0

for i in range(len(unique_words)):
    word = unique_words[i]
    for j in range(len(data)):
        if(Y_train[0][j]==0):
            if word in data[j]:
                ctr1+=1
    for j in range(len(data)):
        if(Y_train[0][j]==1):
            if word in data[j]:
                ctr2+=1
    freqtable.append([word,ctr2,ctr1])
    ctr1 = 0
    ctr2 = 0
    
#Calculating P(yes) and P(No) in the dataset imported
c1 = 0
c2 = 0

for i in Y_train[0]:
    if(i==0):     #NO
        c1+=1
    elif(i==1):   #YES
        c2+=1

pyes = c2/len(Y_train)
pno = c1/len(Y_train)

#compute the conditional probability of each word in the dataset
condprob = list()

yescount = 0
nocount = 0
totalcount = len(freqtable)

for i in range(len(freqtable)):
    if(freqtable[i][1]>0):
        yescount+=1
        
for i in range(len(freqtable)):
    if(freqtable[i][2]>0):
        nocount+=1

posprob = {}
negprob = {}

for i in range(len(freqtable)):
    word = freqtable[i][0]
    val = freqtable[i][1]
    res = (val+1)/(yescount+totalcount)
    posprob.update({word:res})
    
for i in range(len(freqtable)):
    word = freqtable[i][0]
    val = freqtable[i][2]
    res = (val+1)/(nocount+totalcount)
    negprob.update({word:res})
    

#testing the data for input which is not in the trainig set

Y_computed = list()

for i in range(len(X_test)):
    testdata = X_test[i]
    temp = testdata.split(' ')
    uniwords = list()

    for i in temp:
        if i not in uniwords:
            uniwords.append(i)
            
    #for i in uniwords:
        #if(str(i) in stopwords):
            #uniwords.remove(i)
        
    #caluclating probability for test data

    pres = 1
    nres = 1

    for i in uniwords:
        if i in posprob.keys():
            mult = posprob[i]
            pres = pres*mult
    
    pres = pres*pyes

    for i in uniwords:
        if i in negprob.keys():
            mult = negprob[i]
            nres = nres*mult
    nres = nres*pno

    if(pres>nres):
        Y_computed.append(1)
    else:
        Y_computed.append(0)

Y_computed = np.array(Y_computed,dtype=int).reshape(1,len(Y_computed))
counter = 0

for i in range(len(Y_test[0])):
    if(Y_test[0][i]==Y_computed[0][i]):
        counter+=1

per = counter/len(Y_test[0])
per = per * 100
print("Accuracy : ",per)
  
    


            









  




            


    
