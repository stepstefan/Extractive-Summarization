import numpy as np


class Node:
    """Node of the tree"""
    def __init__(self):
        self.left = None
        self.right = None
        self.parent = None
        self.salience = 0
        self.feature = np.array([])
        
