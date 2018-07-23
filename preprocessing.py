import os
from stanfordnlp import StanfordNLP, read_xml


if __name__ == '__main__':
    data1 = u'../baze/DUC2001_Summarization_Documents/docs/'  
    data2 = u'../baze/DUC2002_Summarization_Documents/docs/'  
    data3 = u'../baze/DUC2004_Summarization_Documents/docs/'  

    data = data2    
    c = 0
    for cluster in os.listdir(data):
        docs = data + cluster

        for doc_name in os.listdir(docs):
            doc = docs + '/' + doc_name 

            if doc_name[0:4] == 'FBIS':
                continue

            text = read_xml(doc)
            c += 1

            """
                deo za  racunanje ficera
            """
    print(c)
