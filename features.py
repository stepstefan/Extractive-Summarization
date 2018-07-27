from stanfordcorenlp import StanfordCoreNLP
from stanfordnlp import StanfordNLP
import json
from nltk.tree import Tree
import numpy as np

sNLP = StanfordNLP()

"""Sentences__________________________________________________""" 


class Sentenceftrs:
    def __init__(self, stopwords):
        self.stopwords = stopwords
    def position(self, sentence, slist):
        """The position of the sentences. Suppose there are M sentences in the document, for th ith sentence, the position feature is computed as 1-(i-1)/(M-1)"""
        ith = slist.index(sentence) + 1
        M = len(slist)
        feature = 1 - (ith-1)/(M-1)

        return feature

    def length(self, sentence):
        return len(sentence)

    def subs(self, tree):
        """Sub-sentence count in parsing tree"""
        #t = sNLP.parse(sentence)
        nodes = tree.treepositions()
        cs = 0
        for node in nodes:
            if not type(tree[node]) is str:
                if tree[node].label() == "S" or tree[node].label() == "@S":
                    cs += 1

        return cs

    def depth(self, stree):
        """The root depth of the parsing tree"""
        #tree = sNLP.parse(sentence)
        maxd = 0
        for pos in tree.treepositions():
            if len(pos) > maxd:
                maxd = len(pos)
        return maxd

    def atf(self, sentence, tf_dic):
        """The mean TF values of words in the sentence, devided bu sentence length""" 
        tf_sum = 0 
        wlist = [w.lower() for w in  sNLP.word_tokenize(sentence)]
        for word in wlist:
            tf_sum += tf_dic[word]
        return tf_sum / len(sentence)**2

    def aidf(self, sentence, idf_dic):
        """The mean word IDF values in sentence, devided by the sentence lenght"""
        idf_sum = 0 
        wlist = [w.lower() for w in  sNLP.word_tokenize(sentence)]
        for word in wlist:
            idf_sum += idf_dic[word]
        return idf_sum / len(sentence)**2

    def acf(self, sentence, cf_dic):
        """The mean word CF values in sentence, devided by the sentence length"""
        cf_sum = 0 
        wlist = [w.lower() for w in  sNLP.word_tokenize(sentence)]
        for word in wlist:
            cf_sum += cf_dic[word]
        return cf_sum / len(sentence)**2
        
    def posratio(self, tags):
        """The number of nouns, verbs, adjectives and adverbs in the sentence, devided by sentence length"""
        #tags = sNLP.pos(sentence)
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

    def neration(self, tags):
        """The number of named enitites, devided by sentence length"""
        #tags = sNLP.ner(sentence)
        c1 = 0
        c2 = 0
        for tag in tags:
            c1 += 1
            if tag[1] != "O":
                c2 += 1

        return c2/c1
    
    def numberratio(self, tags):
        """The number of digits, devided by sentence length"""
        #tags = sNLP.pos(sentence)
        c1 = 0
        c2 = 0
        for tag in tags:
            c1 += 1
            if tag[1] == "CD":
                c2 += 1

        return c2/c1
    
    def stopratio(self, wlist):
        """The number of stopwords, devided by sentence length. Use stopword list of ROUGE"""
        counter = 0
        for word in wlist:
            counter += word in self.stopwords
        return counter

"""Wordi features ________________________________________________________________________"""

class Wordftrs:
    """
    word, slist - list of sentences in document
    """
    def __init__(self, idf_dic):
        self.tf_dic = {}
        self.cf_dic = {}
        self.idf_dic = idf_dic

        self.slen_dic = {}

        self.stf_dic = {}
        self.sidf_dic = {}
        self.scf_dic = {}

        self.ss_dic = {}
        self.sd_dic = {}

    def tf(self, slist):
        """ term frequency """ 
        for wlist in slist:
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            for word in wlist:
                if word in self.tf_dic:
                    self.tf_dic[word] += 1
                else:
                    self.tf_dic[word] = 1

    def idf(self, cluster_size):
        """ total document number in the datasets, devided by the frequency of documents which contains the word"""
        pass

    def cf(self, slist):
        """ the frequency of documents which conntains this word in the current cluster"""
        words = []
        for wlist in slist:
            for w in wlist:
                words.append(w.lower())

        wset = set(words)
        for word in wset:
            if word in self.cf_dic:
                self.cf_dic[word] += 1
            else:
                self.cf_dic[word] = 1

    def pos(self, sentence):
        """ a 4-dimension binary vector indicates whether the word is a noun, a verb, an adjective or an adverb. If the word has another part-of-speech, the vector is all-zero"""
        wlist_tuple = sNLP.pos(sentence)
        feature_vec = []

        for wordtuple in wlist_tuple:
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
            feature_vec.append(feature)
        
        return feature_vec

    def namedentity(self, sentence):
        """ a binary value equals one iff the output of named entity classifier from CoreNLP is not entity"""
        feature_vec = []
        for word in sNLP.word_tokenize(sentence):
            ne = sNLP.ner(word)[0]
            if ne[1] == "O":
                feature = 0
            else:
                feature = 1
            feature_vec.append(feature)
            
        return feature_vec

    def number(self, sentence):
        """ a binary value denotes if whether the word is a number"""
        wlist_tuple = sNLP.pos(sentence)
        feature_vec = []

        for wordtuple in wlist_tuple:
            if wordtuple[1] == "CD":
                feature = 1
            else:
                feature = 0
            feature_vec.append(feature)
        return feature_vec
    
    def slen(self, swlist):
        """The maximal length of sentences owning the word"""
        for wlist in swlist:
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            ln = len(wlist)
            for word in wlist:
                if not word in self.slen_dic:
                    self.slen_dic[word] = ln
                else:
                    self.slen_dic[word] = max(self.slen_dic[word], ln)

    def stf(self, slist):
        """The maximal TF score of sentences owning the word"""
        maxes = []  
        for wlist in slist:
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            maxes.append(max([self.tf_dic[wrd] for wrd in wlist]))

        for idx, wlist in enumerate(slist):
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            for word in wlist:
                if not word in self.stf_dic:
                    self.stf_dic[word] = maxes[idx]
                else:
                    self.stf_dic[word] = max(self.stf_dic[word], maxes[idx])

    def scf(self, swlist):
        """The maximal CF score of sentences owning the word"""
        maxes = []  
        for wlist in swlist:
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            maxes.append(max([self.cf_dic[wrd] for wrd in wlist]))

        for idx, wlist in enumerate(swlist):
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            for word in wlist:
                if not word in self.scf_dic:
                    self.scf_dic[word] = maxes[idx]
                else:
                    self.scf_dic[word] = max(self.scf_dic[word], maxes[idx])


    def sidf(self, swlist):
        """The maximal IDF score of sentences owning the word"""
        maxes = []  
        for wlist in swlist:
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            maxes.append(max([self.idf_dic[wrd] for wrd in wlist]))

        for idx, wlist in enumerate(swlist):
            #wlist = [w.lower() for w in sNLP.word_tokenize(sentence)]
            for word in wlist:
                if not word in self.sidf_dic:
                    self.sidf_dic[word] = maxes[idx]
                else:
                    self.sidf_dic[word] = max(self.sidf_dic[word], maxes[idx])

    def update_ss(self, tlist):
        """The maximal sub-sentence count of sentences owning the word. A sub-sentence means the node label is S or @S in parsing tree"""
        for tree in tlist:
            subs = tlist.subs()
            for word in tree.wordlist():
                if not word in self.ss_dic:
                    self.ss_dic[word] = subs
                else:
                    self.ss_dic[word] = max(
                        self.ss_dic[word], subs)

    def update_sd(self, wlist, depth):
        """The maximal parsing tree depth of sentences owning the word"""
        for tree in tlist:
            subs = tlist.depth()
            for word in tree.wordlist():
                if not word in self.ss_dic:
                    self.ss_dic[word] = depth
                else:
                    self.ss_dic[word] = max(
                        self.ss_dic[word], depth)
