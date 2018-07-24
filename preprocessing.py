import os
from stanfordnlp import StanfordNLP, read_xml
from features import Sentenceftrs, Wordftrs

sNLP = StanfordNLP()
wF = Wordftrs()
sF = Sentenceftrs()


if __name__ == '__main__':
    data1 = u'../baze/DUC2001_Summarization_Documents/docs/'  
    data2 = u'../baze/DUC2002_Summarization_Documents/docs/'  
    data3 = u'../baze/DUC2004_Summarization_Documents/docs/'  

    proba = u'./probna_baza/'

    data = proba
    c = 0

    for cluster in os.listdir(data):
        print('Processing cluster: {}'.format(cluster))
        docs = data + cluster

        for doc_name in os.listdir(docs):
            doc = docs + '/' + doc_name 
            print(doc)

            if doc_name[0:4] == 'FBIS':
                continue

            text = read_xml(doc) # <p> tagovi ne rade
            c += 1

            """
                deo za  racunanje ficera
            """
            slist = sNLP.sentances_tokenize(text)
            wF.tf(slist)
            wF.cf(slist)
        wF.idf(len(os.listdir(docs)))
            
    print(c)
