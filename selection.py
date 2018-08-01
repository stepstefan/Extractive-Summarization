from tree import *
import pickle
from scipy import spatial

Tsim = .6

def compare(tree1, tree2):
    vec1 = []
    for word in tree1.getTerminals():
        vec1.append(word.feature[0])
        vec1.append(word.feature[1])

    vec2 = []
    for word in tree2.getTerminals():
        vec2.append(word.feature[0])
        vec2.append(word.feature[1])

    diff = len(vec1) - len(vec2)

    if diff < 0:
        for _ in range(0, -diff):
            vec1.append(0)
    else:
        for _ in range(0, diff):
            vec2.append(0)

    return 1 - spatial.distance.cosine(vec1, vec2)


def greedySelection(pickleF, maxCount):
    treesList = pickle.load( open(pickleF, 'rb') )
    trees = []
    for treeList in treesList:
        for tree in treeList:
            trees.append(tree)
            
    treesSorted = sorted(trees, key=lambda t: t.root.salience)

    wordCount = 0
    summerie = []
    while wordCount < maxCount:
        sentence = treesSorted.pop()
        similarity = 0
        for s in summerie:
            similarity = max(similarity, compare(s, sentence))
        if similarity < Tsim:
            summerie.append(sentence)
            wordCount += len(sentence.wordlist)
    text = ""
    for tree in summerie:
        for word in tree.wordlist:
            text += word + ' '
    return text


if __name__ == '__main__':
    t = greedySelection('d01a.pickle', 100)
    print(t)

