


# Node class for the graph data structure
# Each node has a name, edges, weight, words and their frequency
class Node:
    def __init__(self, id, array_words):
        self.edge = {} # edges connected to the node
        self.id = id # id of the node
        self.words = {}  # words in the node and their frequency
        self.weight = 1 # weight of the node
        # loop for each word in the array
        for word in array_words:
            # count the frequency of each word
            if word in self.words:
                self.words[word] += 1
            else:
                self.words[word] = 1
        # total number of words in the node
        self.num_words = sum(self.words.values())
    
    # connect the node to another node
    def connect(self, node_id, value):
        self.edge[node_id] = value
