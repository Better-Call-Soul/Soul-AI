


# Vertex class for the graph data structure
# Each vertex has a name, edges, weight, words and their frequency
class Vertex:
    def __init__(self, id, array_words):
        self.edge = {} # edges connected to the vertex
        self.id = id # name of the vertex
        self.words = {}  # words in the vertex and their frequency
        self.weight = 1 # weight of the vertex
        # loop for each word in the array
        for word in array_words:
            # count the frequency of each word
            if word in self.words:
                self.words[word] += 1
            else:
                self.words[word] = 1
        # total number of words in the vertex
        self.num_words = sum(self.words.values())
    
    # connect the vertex to another vertex
    def connect(self, vertex_id, value):
        self.edge[vertex_id] = value
