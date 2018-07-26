import numpy as np
from nltk.tree import Tree

def replace(node1, node2):
    node1.label = node2.label
    node1.left = node2.left
    node1.right = node2.right
    node1.start = node2.start
    node1.end = node2.end
    node1.salience = node2.salience
    node1.feature = node2.feature
    node1.isPreTerminal = node2.isPreTerminal
    node1.isTerminal = node2.isTerminal

class Node:
    """Node of the tree"""
    def __init__(self, label, terminal, parent):
        """Init node with label and flag if its terminal node or not"""
        self.left = None
        self.right = None
        self.parent = parent
        self.salience = -1
        self.feature = np.array([])
        self.start = -1
        self.end = -1
        self.label = label
        self.isTerminal = terminal
        self.isPreTerminal = False

    def create(self, tree, wordlist):
        """Create node and its children from tree nltk structure and word list"""
        tpos = tree.treepositions()
        if (0,) in tpos:
            if isinstance(tree[(0,)], str):
                self.left = Node(tree[(0,)], True, self)
                self.isPreTerminal = True
                self.left.start = self.left.end = self.start = self.end = wordlist.index(tree[(0,)])
            else:   
                self.left = Node(tree[(0,)].label(), False, self)
                self.left.create(tree[(0,)], wordlist)
            
        if (1,) in tpos:
            self.right = Node(tree[(1,)].label(), False, self)
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


    def correct(self): 
        """Apply tree corrections at node level"""
        
        # Problem with multiple continous nodes with one child that are not pre-terminal
        if self.label != "ROOT" and self.isPreTerminal is not True:
            print(self.label)
            if self.right is None:
                self.left.correct()
                print("Redirect ", self.label, " to ", self.left.label)
                replace(self, self.left)
            else:
                self.left.correct()
                self.right.correct()




class Stree:
    def __init__(self, tree):
        """Init tree of sentence based on nltk tree"""
        self.root = Node(tree[()].label(), False, None)
        self.wordlist = tree.leaves()
        self.root.create(tree, self.wordlist)

    def correct(self):
        """ Corrects tree for problems with multiple continous nodes with one child and problems with dot at the end of the sentence"""
        
        self.root.left.correct()
            
       
        
