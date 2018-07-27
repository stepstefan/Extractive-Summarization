import os
import time
import numpy as np
import json
from stanfordnlp import StanfordNLP, read_xml
from features import Sentenceftrs, Wordftrs
from tree import *
from nltk.tree import Tree


with open('./duc2002.json', 'r') as f:
    idf_dic = json.load(f)

with open('rouge_155.txt') as f:
    stopwords = [w.strip() for w in f.readlines()]

sNLP = StanfordNLP()
wF = Wordftrs(idf_dic)
sF = Sentenceftrs(stopwords)

def lower_array(a):
    return [x.lower() for x in a]

def read_trees(file_path):
    trees = []
    with open(file_path) as f:
        trees.append(Stree(Tree.fromstring(
            f.readline())).correct())
    return trees



if __name__ == '__main__':
    data1 = u'../baze/DUC2001_Summarization_Documents/docs/'  
    data2 = u'../baze/DUC2002_Summarization_Documents/docs/'  
    data3 = u'../baze/DUC2004_Summarization_Documents/docs/'  

    proba = u'./docs/'
    trees_dic = u'./trees/'

    data = proba
    
    start = time.time()
    for cluster in os.listdir(data):
        print('Processing cluster: {}'.format(cluster))
        docs = data + cluster
        trees_cluster = trees_dic + cluster

        for doc_name in os.listdir(docs):
            doc = docs + '/' + doc_name 
            print(doc)

            if doc_name[0:4] == 'FBIS':
                continue

            text = read_xml(doc)
            trees = read_trees(trees_cluster + '/' + doc_name)

            ### deo za  racunanje ficera ###

            slist = sNLP.sentances_tokenize(text)
            swlist = [sNLP.word_tokenize(x) for x in slist]
            swlist = list(map(lower_array, swlist))

            wF.tf(swlist)
            wF.cf(swlist)

            wF.slen(swlist)

            wF.stf(swlist)
            wF.scf(swlist)

            #wF.update_ss(trees)
            #wF.update_sd(trees)

            for idx, sentence in enumerate(slist):
                tree = sNLP.parse(sentence)
                pos = sNLP.pos(sentence)
                wlist = [x.lower() for x in sNLP.word_tokenize(sentence)]

                _ = wF.pos(sentence) # staviti u tree
                _ = wF.number(sentence) # staviti u tree
                _ = wF.namedentity(sentence) # staviti u tree
                ### Sentence
                _ = sF.position(sentence, slist)
                _ = sF.length(sentence)
                #subs = sF.subs(tree)
                #depth = sF.depth(tree)

                _ = sF.atf(sentence, wF.tf_dic)
                _ = sF.acf(sentence, wF.cf_dic)
                _ = sF.aidf(sentence, wF.idf_dic)

                _ = sF.posratio(pos)
                _ = sF.neration(pos)
                _ = sF.numberratio(pos)
                _ = sF.stopratio(wlist)

                # Word
                
                
    end = time.time()        
    print('Time passed: {} s'.format(int(end - start)))
