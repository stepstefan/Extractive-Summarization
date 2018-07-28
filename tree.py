import numpy as np
from nltk.tree import Tree
from rouge import Rouge

rouge = Rouge()


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

    def depth(self):
        """Depth of tree at node level"""

        if self.isTerminal:
            return 1
        else:
            if self.right is None:
                return 1 + self.left.depth()
            else:
                return 1 + max(self.left.depth(), self.right.depth())

    def subs(self):
        """Sub-sentence count at node level"""
        # print(self.label)
        if self.label == 'S' or self.label == '@S':
            #print(self.label)
            if self.right is None:
                return 1 + self.left.subs()
            else:
                return 1 + self.left.subs() + self.right.subs()
        else:
            if self.left is None:
                return 0
            else:
                if self.right is None:
                    return self.left.subs()
                else:
                    return self.left.subs() + self.right.subs()

    def correct(self): 
        """Apply tree corrections at node level"""
        
        # Problem with multiple continous nodes with one child that are not pre-terminal
        if self.label != "ROOT" and self.isPreTerminal is not True:
            if self.right is None:
                self.left.correct()
                replace(self, self.left)
            else:
                self.left.correct()
                self.right.correct()

    def getTerminals(self):
        """Terminal nodes"""

        if self.isTerminal is True:
            return [self]
        else:
            if self.right is None:
                return self.left.getTerminals()
            else:
                terminals = self.left.getTerminals() + self.right.getTerminals()
                return terminals
    
    def addFeatures(self, features):
        """Adding raw features"""

        terminals = self.getTerminals()
        length = len(terminals)
        if length != len(features):
            print("Mladene nesto si zajebao!")
        else:
            for i in range(length):
                terminals[i].feature = features[i]

    def addSalience(self, wordlist, reference, alpha):
        """Add salience scores on node level"""

        if alpha > 1:
            print("Alpha must be less than or equal 1!")
        else:
            if self.isPreTerminal:
                word = wordlist[self.start]
                if word != ".":
                    self.salience = rouge.get_scores(wordlist[self.start], reference)[0]["rouge-1"]["r"]
                else:
                    self.salience = 0
            else:

                scores = rouge.get_scores('  '.join(wordlist[self.start:self.end+1]), reference)
                self.salience = scores[0]["rouge-1"]["r"]*alpha + scores[0]["rouge-2"]["r"]*(1-alpha)
                if self.left is not None:
                    self.left.addSalience(wordlist, reference, alpha)
                if self.right is not None:
                    self.right.addSalience(wordlist, reference, alpha)

    def traverse(self, function, args):
        """Left traverse tree at node level with function"""

        if self.left is not None:
            self.left.traverse(function, args)
        if self.right is not None:
            self.right.traverse(function, args)
        function(self, args)


class Stree:
    def __init__(self, tree):
        """Init tree of sentence based on nltk tree"""
        self.root = Node(tree[()].label(), False, None)
        self.wordlist = tree.leaves()
        self.root.create(tree, self.wordlist)
        self.sentence_features = np.array([])

    def correct(self):
        """ Corrects tree for problems with multiple continous nodes with one child and problems with dot at the end of the sentence"""
        
        self.root.left.correct()
            
    def getTerminals(self):
        """Get termminal nodes of the tree"""

        return self.root.getTerminals()

    def addFeatures(self, features, sentence_features):
        """Adding raw features ad word and sentance level"""

        self.root.addFeatures(features)
        self.sentence_features = sentence_features

    def addSalience(self, reference, alpha):
        """Adding Salience scores based on reference summaries"""

        self.root.addSalience(self.wordlist, reference, alpha)

    def depth(self):
        """Depth of tree"""

        return self.root.depth()

    def subs(self):
        """Sub-sentence couunt"""

        return self.root.subs()

    def traverse(self, function, args):
        """Left traverse tree with funnction"""

        self.root.traverse(function, args)


def printnode(node, args=None):
    print(node.label)
