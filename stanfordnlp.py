'''
A sample code usage of the python package stanfordcorenlp to access a Stanford CoreNLP server.
Written as part of the blog post: https://www.khalidalnajjar.com/how-to-setup-and-use-stanford-corenlp-server-with-python/ 
'''

from stanfordcorenlp import StanfordCoreNLP
import logging
import json
from nltk.tree import Tree  
import nltk
from xml.etree import ElementTree


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,parse.binaryTrees,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json',
            'parse.binaryTrees': "true"
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def sentances_tokenize(self, text_file):
        return nltk.sent_tokenize(text_file)

    def parse(self, sentence):
        return Tree.fromstring(self.nlp.parse(sentence))

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

def read_xml(file_name):
    name = file_name.split('/')[-1]
    if name[0:4] == 'FBIS':
        return []
    txt = ''
    tree = ElementTree.parse(file_name).getroot()
    if name[0:4] == 'SJMN':
        if not tree.find('LEADPARA') is None:
            txt += tree.find('LEADPARA').text.strip().replace(';', ' ').replace('\n', ' ')
            txt += ' '
        for TEXT in tree.findall('TEXT'):
            txt += TEXT.text.strip().replace(';', ' ').replace('\n', ' ')
        return txt
    elif name[0:3] == 'WSJ':
        if not tree.find('LP') is None:
            txt += tree.find('LP').text.strip().replace('\n', ' ')
            txt += ' '
        for TEXT in tree.findall('TEXT'):
            txt += TEXT.text.strip().replace(';', ' ').replace('\n', ' ')
        return txt
    elif name[0:2] == 'AP' or name[0:2] == 'FT' or name[0:3] == 'NYT':
        for TEXT in tree.findall('TEXT'):
            txt += TEXT.text.strip().replace(';', ' ').replace('\n', ' ')
        return txt
    elif name[0:2] == 'LA':
        for TEXT in tree.findall('TEXT'):
            for p in TEXT.getchildren():
                txt += p.text.strip().replace('\n', ' ')
                txt += ' '
        return txt
    elif name[0] == 'd':
        if tree.getchildren():
            for s in tree.getchildren():
                txt += s.text.strip().replace(';', ' ').replace('\n', ' ') + ' '
        else:
            txt += tree.text.strip().replace(';', ' ').replace('\n', ' ')
        return txt

    else:
        return []
