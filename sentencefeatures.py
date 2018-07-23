from stanfordnlp imprt StanfordNLP
from stanfordcorenlp imprt StanfordCoreNLP
from nltk.tree import Tree
import numpy as np
from wordfeatures import Wordftrs
sNLP = StanfordNLP()
wf=Wordftrs()

class Sentenceftrs:
    def position(self, slist):
        '''The position of the sentences. Suppose there are M sentences in the document, for th ith sentence, the position feature is computed as 1-(i-1)/(M=1)'''
        return 0

    def length(self, sentence):
        return 0

    def subs(self, sentence):
        '''Sub-sentence count in parsing tree'''
        return 0

    def depth(self, sentece):
        '''The root depth of the parsing tree'''
        return 0

    def atf(self, sentence):
        '''The mean TF values of words in the sentence, devided bu sentence length'''   
        return 0

    def aidf(self, sentence):
        '''The mean word IDF values in sentence, devided by the sentence lenght'''
        return 0

    def acf(self, sentence):
        '''The mean word CF values in sentence, devided by the sentence length'''
        return 0
    
    def posratio(self, sentence):
        '''The number of nouns, verbs, adjectives and adverbs in the sentence, devided by sentence length'''
        return 0

    def neration(self, sentence):
        '''The number of named enitites, devided by sentence length'''
        return 0
    
    def numberratio(self, sentence):
        '''The number of digits, devided by sentence length'''
        return 0
    
    def stopratio(self, sentence):
        '''The number of stopwords, devided by sentence length. Use stopword list of ROUGE'''
        return 0
