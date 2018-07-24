from stanfordcorenlp import StanfordCoreNLP
from stanfordnlp import StanfordNLP
import json
from nltk.tree import Tree
import numpy as np

sNLP = StanfordNLP()

'''Sentences__________________________________________________''' 


class Sentenceftrs:
    def position(self, sentence, slist):
        '''The position of the sentences. Suppose there are M sentences in the document, for th ith sentence, the position feature is computed as 1-(i-1)/(M-1)'''
        ith = slist.index(sentence) + 1
        M = len(slist)
        feature = 1 - (ith-1)/(M-1)

        return feature

    def length(self, sentence):
        tokenized = sNLP.word_tokenize(sentence)
        return len(tokenized)

    def subs(self, sentence):
        '''Sub-sentence count in parsing tree'''
        t = sNLP.parse(sentence)
        nodes = t.treepositions()
        cs = 0
        for node in nodes:
            if not type(t[node]) is str:
                if t[node].label() == "S" or t[node].label() == "@S":
                    cs += 1

        return cs

    def depth(self, sentence):
        '''The root depth of the parsing tree'''
        tree = sNLP.parse(sentence)
        maxd = 0
        for pos in tree.treepositions():
            if len(pos) > maxd:
                maxd = len(pos)
        return maxd

    def atf(self, sentence, tf_dic):
        '''The mean TF values of words in the sentence, devided bu sentence length'''   
        tf_sum = 0 
        for word in sentence:
            tf_sum += word[0]
        return tf_sum / len(sentence)**2

    def aidf(self, sentence, idf_dic):
        '''The mean word IDF values in sentence, devided by the sentence lenght'''
        idf_sum = 0 
        for word in sentence:
            idf_sum += word[0]
        return idf_sum / len(sentence)**2

    def acf(self, sentence):
        '''The mean word CF values in sentence, devided by the sentence length'''
        cf_sum = 0 
        for word in sentence:
            cf_sum += word[0]
        return cf_sum / len(sentence)**2
        
    def posratio(self, sentence):
        '''The number of nouns, verbs, adjectives and adverbs in the sentence, devided by sentence length'''
        tags = sNLP.pos(sentence)
        cn = 0
        cv = 0
        cj = 0
        cr = 0
        c = 0
        for tag in tags:
            c += 1
            if tag[1][0] == "N":
                cn += 1
            if tag[1][0] == "V":
                cv += 1
            if tag[1][0] == "J":
                cj += 1
            if tag[1][0:1] == "RB":
                cr += 1
            feature = np.array([cn/c, cv/c, cj/c, cr/c])
        return feature

    def neration(self, sentence):
        '''The number of named enitites, devided by sentence length'''
        tags = sNLP.ner(sentence)
        c1 = 0
        c2 = 0
        for tag in tags:
            c1 += 1
            if tag[1] != "O":
                c2 += 1

        return c2/c1
    
    def numberratio(self, sentence):
        '''The number of digits, devided by sentence length'''
        tags = sNLP.pos(sentence)
        c1 = 0
        c2 = 0
        for tag in tags:
            c1 += 1
            if tag[1] == "CD":
                c2 += 1

        return c2/c1
    
    def stopratio(self, sentence):
        '''The number of stopwords, devided by sentence length. Use stopword list of ROUGE'''
        return 0


sf = Sentenceftrs()

'''Wordi features ________________________________________________________________________'''

class Wordftrs:
    """
    word, slist - list of sentences in document
    """

    tf_dic = {}
    idf_dic = {}
    cf_dic = {}

    def tf(self, slist):
        ''' term frequency '''
        for sentence in slist:
            wlist = [w.lower() for w in  sNLP.word_tokenize(sentence)]
            for word in wlist:
                if word in self.tf_dic:
                    self.tf_dic[word] += 1
                else:
                    self.tf_dic[word] = 1

    def idf(self, cluster_size):
        ''' total document number in the datasets, devided by the frequency of documents which contains the word'''
        for key in list(self.cf_dic.keys()):
            self.idf_dic[key] = cluster_size / self.cf_dic[key]

    def cf(self, slist):
        ''' the frequency of documents which conntains this word in the current cluster'''
        wlist = []
        for sentence in slist:
            for w in sNLP.word_tokenize(sentence):
                wlist.append(w.lower())

        wset = set(wlist)
        for word in wset:
            if word in self.cf_dic:
                self.cf_dic[word] += 1
            else:
                self.cf_dic[word] = 1

    def pos(self, wordtuple):
        ''' a 4-dimension binary vector indicates whether the word is a noun, a verb, an adjective or an adverb. If the word has another part-of-speech, the vector is all-zero'''
        if wordtuple[1][0] == "N":
            feature = np.array([1, 0, 0, 0])
        elif wordtuple[1][0] == "V":
            feature = np.array([0, 1, 0, 0])
        elif wordtuple[1][0] == "J":
            feature = np.array([0, 0, 1, 0])
        elif wordtuple[1][0:1] == "RB":
            feature = np.array([0, 0, 0, 1])
        else:
            feature = np.array([0, 0, 0, 0])
        
        return feature

    def namedentity(self, word):
        ''' a binary value equals one iff the output of named entity classifier from CoreNLP is not entity'''
        ne = sNLP.ner(word)[0]
        if ne[1] == "O":
            feature = 0
        else:
            feature = 1
            
        return feature

    def number(self, wordtuple):
        ''' a binary value denotes if whether the word is a number'''
        if wordtuple[1] == "CD":
            feature = 1
        else:
            feature = 0
        return feature
    
    def slen(self, word, slist):
        '''The maximal length of sentences owning the word'''
        word = word.lower()
        maximal = 0
        for sentence in slist:
            wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            if (word.lower() in wlist):
                maximal = max(maximal, len(wlist))
        return maximal

    def stf(word, claster, tf_dic):
        '''The maximal TF score of sentences owning the word'''
        maximal = 0
        for slist in claster:
            for sentence in slist:
                wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
                if word.lower() in wlist:
                    mx = max([tf_dic[wrd] for wrd in wlist])
                    maximal = max(maximal, mx)
        return maximal
                
    def scf(word, claster, cf_dic):
        '''The maximal CF score of sentences owning the word'''
        maximal = 0
        for slist in claster:
            for sentence in slist:
                wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
                if word.lower() in wlist:
                    mx = max([cf_dic[wrd] for wrd in wlist])
                    maximal = max(maximal, mx)
        return maximal


    def sidf(word, claster, idf_dic):
        '''The maximal IDF score of sentences owning the word'''
        maximal = 0
        for slist in claster:
            for sentence in slist:
                wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
                if word.lower() in wlist:
                    mx = max([idf_dic[wrd] for wrd in wlist])
                    maximal = max(maximal, mx)
        return maximal

    def ssubs(self, word, slist):
        '''The maximal sub-sentence count of sentences owning the word. A sub-sentence means the node label is S or @S in parsing tree'''
        maxs = 0
        for sen in slist:
            wlist = [w.lower() for w in sNLP.word_tokenize(sen)]
            if word.lower() in wlist:
                subcount = sf.subs(sen)
                maxs = max(maxs, subcount)
        return maxs

    def sdepth(self, word, slist):
        '''The maximal parsing tree depth of sentences owning the word'''
        maxd = 0
        for sen in slist:
            wlist = [w.lower() for w in sNLP.word_tokenize(sen)]
            if word.lower() in wlist:
                dep = sf.depth(sen)
                maxd = max(maxd, dep)
        return maxd


