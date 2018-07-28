import os
import time
import numpy as np
import pickle
import json
from stanfordnlp import StanfordNLP, read_xml
from features import Sentenceftrs, Wordftrs
from tree import *
from nltk.tree import Tree


with open('./idf.json', 'r') as f:
    idf_dic = json.load(f)
    for word in idf_dic:
        idf_dic[word] = len(idf_dic) / idf_dic[word]

with open('rouge_155.txt') as f:
    stopwords = [w.strip() for w in f.readlines()]

sNLP = StanfordNLP()
wF = Wordftrs(idf_dic)
sF = Sentenceftrs(stopwords)


def gen_sen(tree):
    sentece = ""
    for w in tree.wordlist:
        sentece += w + ' '
    return sentece[:-1]


def read_trees(file_path):
    trees = []
    with open('log', 'w') as log:
        with open(file_path) as f:
            text = f.readlines()
        for sentace in text:
            log.write(sentace)
            t = Stree(Tree.fromstring(
                sentace))
            t.correct()
            trees.append(t)
        return trees


if __name__ == '__main__':

    ###########
    trees_dic = u'./trees2002/'
    end_location = u'./parsed_trees2002/'
    ###########

    data = trees_dic
    
    start_whole = time.time()
    for idx, cluster in enumerate(os.listdir(data)):
        start = time.time()
        print('Processing cluster: {} ({}/{})'.format(cluster, idx+1, len(os.listdir(data))))
        trees_cluster = trees_dic + cluster

        for doc_name in os.listdir(trees_cluster):

            if doc_name[0:4] == 'FBIS':
                continue

            trees = read_trees(trees_cluster + '/' + doc_name)
            
            swlist = []
            for tree in trees:
                swlist.append([w.lower() for w in tree.wordlist])

            # deo za  racunanje ficera ###

            wF.tf(swlist)
            wF.cf(swlist)

            wF.slen(swlist)

            wF.stf(swlist)
            wF.scf(swlist)
            wF.sidf(swlist)

            wF.update_ss(trees)
            wF.update_sd(trees)

        cluster_pickle = []
        print('Writing features . . .')
        for doc_name in os.listdir(trees_cluster):
            tree_list = []
            trees = read_trees(trees_cluster +
                    '/' + doc_name)

            for tree in trees:
                sentence = gen_sen(tree)
                wlist_tuple = sNLP.pos(sentence)
                wlist = [w.lower() for w in tree.wordlist]

                pos = wF.pos(wlist_tuple)
                number = wF.number(sentence)
                ne = wF.namedentity(sentence) 

                ### Sentence
                position = sF.position(tree, trees)
                length = sF.length(sentence)
                subs = tree.subs()
                depth = tree.depth()

                atf = sF.atf(sentence, wF.tf_dic)
                acf = sF.acf(sentence, wF.cf_dic)
                aidf = sF.aidf(sentence, wF.idf_dic)

                posratio = sF.posratio(wlist_tuple)
                neration = sF.neration(wlist_tuple)
                numberratio = sF.numberratio(wlist_tuple)
                stopratio = sF.stopratio(wlist)

                tree_ftrs = []
                wlist = [w.lower() for w in tree.wordlist]
                for idx, word in enumerate(wlist):
                    word_ftrs = np.array([
                        wF.tf_dic[word],
                        wF.idf_dic[word],
                        wF.cf_dic[word],
                        pos[idx],
                        ne[idx],
                        number[idx],
                        wF.slen_dic[word],
                        wF.stf_dic[word],
                        wF.scf_dic[word],
                        wF.sidf_dic[word],
                        wF.ss_dic[word],
                        wF.sd_dic[word],
                        ], dtype=object)
                    tree_ftrs.append(word_ftrs)
                sen_ftrs = np.array([
                    position,
                    length,
                    subs,
                    depth,
                    sF.atf,
                    sF.aidf,
                    sF.acf,
                    posratio,
                    neration,
                    numberratio,
                    stopratio,
                    ])
                tree.addFeatures(tree_ftrs, sen_ftrs)
            cluster_pickle.append(trees)
        
        print('Pickleing {} cluster!'.format(cluster))
        with open(end_location + cluster + '.pickle', 'wb') as p:
            pickle.dump(cluster_pickle, p)
                
        end = time.time()        
        print('Time passed: {} s'.format(int(end - start)))
    end_whole = time.time()
    print('Total time passed {} s'.format(int(end_whole - start_whole)))
