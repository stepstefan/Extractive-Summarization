from stanfordcorenlp import StanfordCoreNLP
import json
from nltk.tree import Tree
from stanfordnlp import StanfordNLP
import numpy as np


sNLP = StanfordNLP()

class Wordftrs:

# word, slist - list of sentences in document

    def tf(self, word, slist):
        ''' term frequency '''
        word = word.lower()
        counter = 0
        for sentence in slist:
            wlist = [w.lower() for w in  sNLP.word_tokenize(sentence)]
            for wrd in wlist:
                counter += (word == wrd)
        return counter

    def idf(word, slist):
        ''' total document number in the datasets, devided by the frequency of documents which contains the word'''
        pass

    def cf(word, slist):
        ''' the frequency of documents which conntains this word in the current cluster'''
        return 0

    def pos(self, wordtuple):
        ''' a 4-dimension binary vector indicates whether the word is a noun, a verb, an adjective or an adverb. If the word has another part-of-speech, the vector is all-zero'''
        if wordtuple[1][0]=="N":
            feature = np.array([1,0,0,0])
        elif wordtuple[1][0]=="V":
            feature = np.array([0,1,0,0])
        elif wordtuple[1][0]=="J":
            feature = np.array([0,0,1,0])
        elif wordtuple[1][0]=="R":
            feature = np.array([0,0,0,1])
        else:
            feature = np.array([0,0,0,0])
        
        return feature

    def namedentity(self, word):
        ''' a binary value equals one iff the output of named entity classifier from CoreNLP is not entity'''
        ne=sNLP.ner(word)[0]
        if ne[1]=="O":
            feature=0
        else:
            feature=1
            
        return feature

    def number(self, wordtuple):
        ''' a binary value denotes if whether the word is a number'''
        if wordtuple[1]=="CD":
            feature = 1
        else:
            feature = 0
        return feature
    
    def slen(self, word, slist):
        '''The maximal length of sentences owning the word'''
        word = word.lower()
        maximal = 0
        for sentence in slist:
            wlist = [w.lower() for w in  sNLP.word_tokenize(sentence)]
            if (word in wlist):
                maximal = max(maximal, len(wlist))
        return maximal

    def stf(word, slist):
        '''The maximal TF score of sentences owning the word'''
        pass

    def scf(word, slist):
        '''The maximal CF score of sentences owning the word'''
        return 0

    def sidf(word, slist):
        '''The maximal IDF score of sentences owning the word'''
        return 0
    def ssubs(word, slist):
        '''The maximal sub-sentence count of sentences owning the word. A sub-sentence means the node label is S or @S in parsing tree'''
        return 0
    def sdepth(word, slist):
        '''The maximal parsing tree depth of sentences owning the word'''
        return 0





