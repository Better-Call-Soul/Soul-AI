import re
import math
from nltk.tokenize import word_tokenize
from preprocess.preprocess import Preprocessor
from preprocess.fastcoref import Fastcoref
from preprocess.capitalize import Capitalize
from vectorizers.tf_idf_vectorizer import TFIDFVectorizer
from vectorizers.mini_lm import MiniLM
from data_structures.node import Node
from preprocess.capitalize import Capitalize
from preprocess.preprocess import Preprocessor
from preprocess.fastcoref import Fastcoref
from sentence_transformers import  util
from transformer.roberta_base_go_emotions import Classifier
from utils.utils import best_len_of_summary
from typing import Literal
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# TextRank
# This class is used to summarize text using the TextRank algorithmd
class TextRank:
    def __init__(self,embeddings_type: Literal['tf-idf', 'mini-lm']='mini-lm',
                method: Literal['jacard-index', 'cosine-similarity']='cosine-similarity',dampening_factor:int=0.85,error_threshold:int=0.01):
        self.nodes = []
        self.dampening_factor = dampening_factor # damping factor used in the PageRank algorithm
        self.num_nodes = 0
        self.org_sentences = []
        self.filter_sentences = []
        self.error_threshold =error_threshold # error threshold used to stop the iteration
        self.unconnected_nodes = set()
        self.fastcoref=Fastcoref()
        self.capitalize=Capitalize()
        self.preprocess=Preprocessor()
        self.embeddings_type=embeddings_type
        self.method=method
        # check the embeddings type
        if(embeddings_type=='tf-idf'):
            self.vectorizer = TFIDFVectorizer()
        elif(embeddings_type=='mini-lm'):
            self.vectorizer = MiniLM()
        else:
            raise ValueError("Invalid embeddings type")
        self.classifier=Classifier()
        self.transform_flag=True
    
    # calculate the jacard index between two sentences
    def jaccard_index(self,node1:Node, node2: Node)->float:
        # get sentence embeddings
        sentence1 = self.filter_sentences[node1.id]
        sentence2 = self.filter_sentences[node2.id]
        
        # tokenize the sentences into words using NLTK
        set1 = set(word_tokenize(sentence1))
        set2 = set(word_tokenize(sentence2))
        
        # calculate the intersection and union of the sets
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        # compute the Jaccard index
        jaccard_sim = len(intersection) / len(union)
        
        return jaccard_sim

    # initialize the nodes with the cosine similarity between the sentences
    def initialize(self):
        for node_num in range(len(self.nodes) - 1):
            for inner_vertex in range(node_num + 1, len(self.nodes)):
                if(self.method=='jacard-index'):
                    # calculate the jacard index between the sentences
                    similarity = self.jaccard_index(self.nodes[node_num], self.nodes[inner_vertex])
                elif(self.method=='cosine-similarity'):
                    # calculate the cosine similarity between the sentences
                    similarity = self.cosine_similarity_nodes(self.nodes[node_num], self.nodes[inner_vertex])
                else:
                    raise ValueError("Invalid method")
                self.nodes[node_num].connect(inner_vertex, similarity)
                self.nodes[inner_vertex].connect(node_num, similarity)
    
    # # get the best length of the summary
    # def best_len_of_summary(self)->int:
    #     if len(self.nodes) <= 3:
    #         return len(self.nodes)
    #     return round(1.3 * math.log(len(self.org_sentences)))
    
    # rank the nodes based on the weight
    def rank_nodes(self)->list[Node]:
        self.nodes.sort(key=lambda node: node.weight, reverse=True)
    
    # process the text data and create the nodes
    def process(self, text:str):
        # clear
        self.num_nodes = 0
        self.nodes=[]
        self.unconnected_nodes = set()
        self.filter_sentences = []  
        self.org_sentences = []
        
        
        text=self.fastcoref.coreference_resolution(text)
        # split the text into sentences and preprocess each sentence
        statements = re.findall(r'[^.!?]+[.!?]', text)
        # use a set to track seen statements and a list to store unique statements
        seen_statements = set()
        unique_statements = []

        for statement in statements:
            # remove leading/trailing whitespace and convert to lowercase for comparison
            clean_statement = statement.strip().lower()

            # if the statement is not already seen, add to unique list and seen set
            if clean_statement not in seen_statements:
                unique_statements.append(statement.strip())
                seen_statements.add(clean_statement)
        statements=unique_statements
        self.org_sentences =  [statement.strip() for statement in statements]
        # preprocessing steps:
        # print("statements",statements)
        for sentence in statements:
            sentence=self.preprocess.clean(sentence,
            ["lower_sentence","remove_emojis","remove_emoticons","remove_nonascii_diacritic",
            "remove_emails","clean_html",
            "remove_url","replace_repeated_chars","expand_sentence","remove_non_alphabetic",
            "remove_extra_space","tokenize_sentence","remove_stop_words","lemm_sentence",
            "check_sentence_spelling","detokenize_sentence"]
            ,"")[0]
            # tokenize the sentence
            word_list = word_tokenize(sentence)
            self.filter_sentences.append(sentence)
            # add the node to the graph
            self.add_node(word_list)
    
    # add a node to the graph
    def add_node(self, word_list: list[str]):
        self.nodes.append(Node(self.num_nodes, word_list))
        self.num_nodes += 1
    
    # calculate the cosine similarity between two nodes
    def cosine_similarity_nodes(self, node1:Node, node2: Node)->float:
        # get sentence embeddings 
        sentence1 = self.filter_sentences[node1.id]
        sentence2 = self.filter_sentences[node2.id]
        if(self.embeddings_type=='mini-lm'):
            embedding1 = self.vectorizer.encode(sentence1)
            embedding2 = self.vectorizer.encode(sentence2)
            # compute cosine similarity
            similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
            return similarity
        elif (self.embeddings_type=='tf-idf'):
            embedding=self.vectorizer.fit_transform([sentence1,sentence2])
            embedding1=embedding[0]
            embedding2=embedding[1]
            # compute the dot product
            dot_product = np.dot(embedding1, embedding2)

            # compute the magnitudes (norms) of the vectors
            norm_1 = np.linalg.norm(embedding1)
            norm_2 = np.linalg.norm(embedding2)
            # compute the cosine similarity
            # compute cosine similarity
            similarity = dot_product / (norm_1 * norm_2)
            return similarity


    
    # get the total edge weights for each node
    def get_edge_totals(self)->list[int]:
        # get the total edge weights for each node
        out_edges = []
        for node_num in range(self.num_nodes):
            temp_sum = sum(self.nodes[node_num].edge.values())
            # if the node is not connected to any other node
            if not temp_sum:
                self.nodes[node_num].weight = 0
                self.unconnected_nodes.add(node_num)
            # add the total edge weights to the list
            out_edges.append(temp_sum)
        # return the list of total edge weights
        return out_edges
    
    # update the weights of the nodes
    def update_nodes_weights(self):
        # get the total edge weights for each node
        total_edge_weights = self.get_edge_totals()
        # update the weights of the nodes
        for vertex_num in range(self.num_nodes):
            temp_sum = 0
            # calculate the new weight of the node
            for other_vertex in range(self.num_nodes):
                if other_vertex != vertex_num and other_vertex not in self.unconnected_nodes:
                    temp_sum += ((self.nodes[vertex_num].edge[other_vertex] / total_edge_weights[other_vertex]) * self.nodes[other_vertex].weight)
            # update the weight of the node
            self.nodes[vertex_num].weight = ((1 - self.dampening_factor) + self.dampening_factor * temp_sum)
            # if the transform flag is set
            if(self.transform_flag):
                self.nodes[vertex_num].weight+=0.2*self.classifier.mental_health_score(self.classifier.predict(self.org_sentences[self.nodes[vertex_num].id])[0])
            # print(self.nodes[vertex_num].weight)
    
    # get the top sentences
    def get_top_sentences(self, num_sentences:int)->str:
        self.rank_nodes()
        # print("num_sentences",num_sentences)
        top_sentences_arr = []
        if num_sentences > self.num_nodes:
            return ' '.join(self.org_sentences)
        else:
            # get the top sentences
            for sentence_out in range(num_sentences):
                top_sentences_arr.append(self.nodes[sentence_out])
            # sort the sentences by their id
            top_sentences_arr.sort(key=lambda vert: vert.id)
            # get the summary statements
            summary_statements = [self.org_sentences[vert.id] for vert in top_sentences_arr]
            return ' '.join(summary_statements)

    # summarize the text data
    def summary(self, text:str, num_summary_sentences:int=None)->str:

        # process the text data
        self.process(text)
        # initialize the nodes
        self.initialize()
        # initialize the last and current weights
        last= None
        cur = None
        # set the error level
        error_level = 10
        # iterate until the error level is less than the threshold
        while error_level > self.error_threshold:

            self.update_nodes_weights()
            if cur is None:
                # cur = self.nodes[0].weight
                # get the sum of the weights of the nodes and divide by the number of nodes
                cur=sum([node.weight for node in self.nodes])/len(self.nodes)
            else:
                last = cur
                # cur = self.nodes[0].weight
                cur=sum([node.weight for node in self.nodes])/len(self.nodes)
                error_level = abs(cur - last)
            
                
        # get the best length of the summary if not provided
        if not num_summary_sentences:
            num_summary_sentences = best_len_of_summary(self.org_sentences)
        
        # get the top sentences
        output = self.get_top_sentences(num_summary_sentences)

        return output
