from stanfordcorenlp import StanfordCoreNLP
import json
from nltk.tree import Tree
from stanfordnlp import StanfordNLP
import re


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

    def pos(word):
        ''' a 4-dimension binary vector indicates whether the word is a noun, a verb, an adjective or an adverb. If the word has another part-of-speech, the vector is all-zero'''
        return 0

    def namedentity(word):
        ''' a binary value equals one iff the output of named entity classifier from CoreNLP is not entity'''
        return 0

    def number(word):
        ''' a binary value denotes if whether the word is a number'''
        return 0
    
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





