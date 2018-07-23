from stanfordnlp import StanfordNLP
from stanfordcorenlp import StanfordCoreNLP
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
        tokenized = sNLP.word_tokenize(sentence)
        return len(tokenized)

    def subs(self, sentence):
        '''Sub-sentence count in parsing tree'''
        return 0

    def depth(self, sentence):
        '''The root depth of the parsing tree'''
        tree=sNLP.parse(sentence)
        maxd = 0
        for pos in tree.treepositions():
            if len(pos)>maxd:
                maxd = len(pos)
        return maxd

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
        tags = sNLP.pos(sentence)
        cn=0;cv=0;cj=0;cr=0;c=0
        for tag in tags:
            c+=1
            if tag[1][0]=="N":
                cn+=1
            if tag[1][0]=="V":
                cv+=1
            if tag[1][0]=="J":
                cj+=1
            if tag[1][0:1]=="RB":
                cr+=1
            feature = np.array([cn/c,cv/c,cj/c,cr/c])
        return feature

    def neration(self, sentence):
        '''The number of named enitites, devided by sentence length'''
        tags = sNLP.ner(sentence)
        c1=0;c2=0
        for tag in tags:
            c1+=1
            if tag[1]!="O":
                c2+=1

        return c2/c1
    
    def numberratio(self, sentence):
        '''The number of digits, devided by sentence length'''
        tags = sNLP.pos(sentence)
        c1=0;c2=0
        for tag in tags:
            c1+=1
            if tag[1]=="CD":
                c2+=1

        return c2/c1
    
    def stopratio(self, sentence):
        '''The number of stopwords, devided by sentence length. Use stopword list of ROUGE'''
        return 0
