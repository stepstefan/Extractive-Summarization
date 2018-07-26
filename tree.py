import numpy as np
from nltk.tree import Tree


class Node:
    """Node of the tree"""
    def __init__(self, label, terminal):
        self.left = None
        self.right = None
        self.parent = None
        self.salience = -1
        self.feature = np.array([])
        self.start = -1
        self.end = -1
        self.label = label
        self.isTerminal = terminal
        self.isPreTerminal = False

    def create(self, tree, wordlist):
        tpos = tree.treepositions()
        if (0,) in tpos:
            if isinstance(tree[(0,)], str):
                self.left = Node(tree[(0,)], True)
                self.isPreTerminal = True
                self.left.start = self.left.end = self.start = self.end = wordlist.index(tree[(0,)])
            else:   
                self.left = Node(tree[(0,)].label(), False)
                self.left.create(tree[(0,)], wordlist)
            
        if (1,) in tpos:
            self.right = Node(tree[(1,)].label(), False)
            self.right.create(tree[(1,)], wordlist)

        if self.left is not None and self.right is not None:
            self.start = min(self.left.start, self.right.start)
            self.end = max(self.left.end, self.right.end)
        elif self.left is not None and self.right is None:
            self.start = self.left.start
            self.end = self.left.end
        elif self.left is None and self.right is not None:
            self.start = self.right.start
            self.end = self.right.end


class Stree:
    def __init__(self, tree):
        self.root = Node(tree[()].label(), False)
        self.wordlist = tree.leaves()
        self.root.create(tree, self.wordlist)
    
    def correct(self):
        """ Corrects tree for problems with multiple continous nodes with one child and problems with dot at the end of the sentence"""
        
        # Problem with multiple continnous nodes with one child that are not pre-terminal

       
        
